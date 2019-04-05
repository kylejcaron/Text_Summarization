import pandas as pd
import matplotlib.pyplot as plt
import pyspark
import numpy as np
import string
import re
from pyspark.sql.functions import isnan, when, count, col
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql import Row
import pyspark.sql.functions as F
import pickle
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
addl_punctuation = set(['...', '`', '¿','⸮'])
PUNCTUATION = PUNCTUATION.union(addl_punctuation)


CONTRACTIONS = { 
"ain't": "am not", "aren't": "are not", "can't": "cannot","can't've": "cannot have","'cause": "because",
"could've": "could have","couldn't": "could not","couldn't've": "could not have","didn't": "did not",
"doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have",
"hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have",
"he'll": "he will","he's": "he is","how'd": "how did","how'll": "how will",
"how's": "how is","i'd": "i would","i'll": "i will","i'm": "i am","i've": "i have","isn't": "is not",
"it'd": "it would","it'll": "it will","it's": "it is","let's": "let us", "ma'am": "madam", "mayn't": "may not",
"might've": "might have","mightn't": "might not","must've": "must have","mustn't": "must not",
"needn't": "need not","oughtn't": "ought not","shan't": "shall not","sha'n't": "shall not","she'd": "she would",
"she'll": "she will","she's": "she is","should've": "should have","shouldn't": "should not","that'd": "that would",
"that's": "that is","there'd": "there had","there's": "there is","they'd": "they would","they'll": "they will",
"they're": "they are","they've": "they have","wasn't": "was not","we'd": "we would","we'll": "we will",
"we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what're": "what are",
"what's": "what is","what've": "what have","where'd": "where did","where's": "where is","who'll": "who will",
"who's": "who is","won't": "will not","wouldn't": "would not","you'd": "you would","you'll": "you will",
"you're": "you are"
}

def clean_text(text, remove_stopwords=True):
    text = text.lower()
    
    if True:
        text = text.split()
        new_text = [CONTRACTIONS[w] if w in CONTRACTIONS else w for w in text]
        text = " ".join(new_text)
    
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in PUNCTUATION]
    if remove_stopwords==True:
        tokens = [w for w in tokens if w not in STOPWORDS]
    return tokens

def word_length(string):
    tokens = word_tokenize(string)
    tokens = [w for w in tokens if w not in PUNCTUATION]
    return len(tokens)

def clean_data(df, n_words_summary=50, remove_stopwords=True):
    # Get rid of all rows where subreddit is null (these are spam)
    df = df.filter(df.subreddit.isNotNull())
    # Lowercase columns:
    for col in ['body','content','normalizedBody','subreddit','summary','title']:
        df = df.withColumn(col, F.lower(F.col(col)))
    # Converts 'null' strings in the title column back to null values
    df = df.withColumn('title', when(df.title == 'null', F.lit(None)).otherwise(df.title))    
    
    # Creat edit(bool) and edit_len columns, while removing 'edit:%' from summary column
    split_col = F.split(df['summary'], '(edit:|[^a-z]edit)')
    df = df.withColumn('edit', split_col.getItem(1))
    df = df.withColumn('summary', split_col.getItem(0))
    function = udf(word_length, LongType())
    df = df.withColumn('summary_len', function(df.summary))
        # Creates edit_len column, number of words from 'edit'
    df = df.withColumn('edit', df.edit).na.fill('')
    df = df.withColumn('edit_len', function(df.edit))
        # Converts -1 in edit_len column to null
    df = df.withColumn('edit_len',
        when(df.edit_len == -1, F.lit(0)).otherwise(df.edit_len))
    df = df.withColumn('edit', when(df.edit.isNull(), F.lit(0)).otherwise(1))
    # Remove all rows where summary contains less than 5 words
    df = df.filter(df.summary_len >= 5)
    # Remove all rows where summary contains greater than n_words_summary words
    df = df.filter((df.summary_len <= n_words_summary))
    # Remove all rows where the summary length is not less than 50% of the content length
    df = df.filter(df.summary_len <= df.content_len*0.5)
    # Clean Content column
    cleantext_udf = udf(clean_text, StringType())
    df = df.withColumn('content', cleantext_udf(df.content, F.lit(remove_stopwords)))
    df = df.withColumn('summary', cleantext_udf(df.summary, F.lit(False)))
    return df
