import pandas as pd
import matplotlib.pyplot as plt
import pyspark
import numpy as np
import string
import re
from pyspark.sql.functions import isnan, when, count, col
import pyspark.sql.functions as F

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, IntegerType, FloatType, DoubleType, LongType
from pyspark.sql import Row
import pickle
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
from pyspark.sql import SparkSession


def word_length(string):
    tokens = word_tokenize(string)
    tokens = [w for w in tokens if w not in PUNCTUATION]
    return len(tokens)

def clean_text(text, remove_stopwords=True):
    text = text.lower()
    
    if True:
        text = text.split()
        new_text = [CONTRACTIONS[w] if w in CONTRACTIONS else w for w in text]
        text = " ".join(new_text)
    
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in PUNCTUATION]
    if remove_stopwords==True:
        tokens = [w for w in tokens if w not in STOPWORDS]
    return tokens

def clean_df(df, output_csv=False):

    # Lowercase reviews and summaries
    for col in ['Summary', 'Text']:
        df = df.withColumn(col, F.lower(F.col(col)))
    # Create word_length columns    
    function = udf(word_length, LongType())
    df = df.withColumn('Summary_len', function(df.Summary))
    df = df.withColumn('Text_len', function(df.Text))    
    # Filter rows by word lengths
    df = df.filter(df.Summary_len <= 20)
    df = df.filter(df.Text_len > 20)
    df = df.filter(df.Text_len*0.5 > df.Summary_len)
    df = df.filter(df.Text_len<=250)
    # Clean text
    cleantext_udf = udf(clean_text, StringType())
    df = df.withColumn('Summary_cleaned', 
            cleantext_udf(df.Summary, F.lit(True)))
    # Clean reviews
    df = df.filter(df.Score.isin(['1','2','3','4','5']))
    df = df.withColumn("Score", df["Score"].cast(IntegerType()))
    if output_csv==True:
        subset = df.select('Text', 'Summary')
        train, test = subset.randomSplit([0.9,0.1])
        X_train, y_train = train.select('Text'), train.select('Summary')
        X_test, y_test = test.select('Text'), test.select('Summary')
        X_train.write.csv('sumdata/train/train.article.csv')
        y_train.write.csv('sumdata/train/train.title.csv')
        X_test.write.csv('sumdata/train/valid.article.filter.csv')
        y_test.write.csv('sumdata/train/talid.title.filter.csv')
    return df

def sentiment_subset(df):
    subset_1star = df.filter(df.Score == 1)
    subset_5star = df.filter(df.Score == 5)
    # Random undersampling
    sample_ratio = subset_1star.count()/subset_5star.count()
    subset_5star = subset_5star.sample(withReplacement=False, fraction=sample_ratio)
    subset = subset_1star.union(subset_5star)
    return subset

if __name__ == "__main__":
    spark = SparkSession.builder.appName("CleaningApp").getOrCreate()
    df = spark.read.csv(
    's3n://aws-logs-816063959671-us-east-1/data/Reviews.csv',header=True)
    df.cache()
    clean_df(df, output_csv=True)

