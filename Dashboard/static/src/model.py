import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))


class Model(object):
	def __init__(self):
		# self.model = GradientBoostingClassifier(learning_rate=0.01, max_depth=8, 
		# 	max_features=5, min_samples_leaf=5, n_estimators=1500)
		self.model = MultinomialNB()
		self.tfidf = TfidfVectorizer(max_df=0.95, min_df=2,
                                   stop_words='english', 
                                   lowercase=True)
		pass
	
	def fit(self, X, y):
		# Import X and y as text
		X = self.tfidf.fit_transform(X)
		y = y
		self.model.fit(X, y)
		filename = 'data/model.pkl'
		pickle.dump(self, open(filename, 'wb')) 
		return self
	
	def predict(self, X):
		X = self.tfidf.transform(X)
		predictions = self.model.predict(X)
		return predictions

	def predict_proba(self,X):
		X = self.tfidf.transform(X)
		proba_predictions = self.model.predict_proba(X)
		return proba_predictions

	def score(self, X, y):
		X = self.tfidf.transform(X)
		score = self.model.score(X, y)
		return score


def get_data():
   client_name = 'fraud_db'
   tab_name = 'events'
   #mongo_cols = {'acct_type','user_type','email_domain','venue_state','venue_name'}
   client = MongoClient()
   db = client[client_name]
   tab = db[tab_name]
   cursor = tab.find(None) #mongo_cols)
   df = pd.DataFrame(list(cursor))
   return df

def clean_text(text, remove_stopwords=True, rejoin=True):
    text = text.lower()
    # Remove tags and characters
    cleanr = re.compile("<.*?>|[*`\'\"+=.]|http\S+|href|span|class")
    text = re.sub(cleanr, '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in PUNCTUATION]
    if remove_stopwords==True:
        tokens = [w for w in tokens if w not in STOPWORDS]
    if rejoin==True:
        tokens = ' '.join(tokens)
    return tokens


if __name__ == '__main__':
	# read data
	
	df = pd.read_csv('data/sentiment_analysis_subset.csv')\
	            .drop(['Unnamed: 0', 'Id'],axis=1)
	print('cleaning....')
	df.Summary = df.Summary.apply(clean_text, args=(False,True))
	df.Text = df.Text.apply(clean_text, args=(True,True))
	text_df = df[['Text', 'Summary', 'Score']]
	text_df.Score = text_df.Score.replace(1,0)
	text_df.Score = text_df.Score.replace(5,1)
	X = text_df['Text']
	y = text_df['Score']

	X_train, X_test, y_train, y_test = \
	    train_test_split(X, y, train_size=0.75, shuffle=True, stratify=y)

	print('Fitting....')
	#fit model
	model = Model()
	model.fit(X_train, y_train)

	print('score: {}'.format(model.score(X_test,y_test)))
	print('Test_Proba {}'.format(model.predict_proba(X_test)))


	



