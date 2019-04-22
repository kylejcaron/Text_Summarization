import numpy as np
import pandas as pd
from pymongo import MongoClient
from api_client import EventAPIClient


def get_data():
   client_name = 'fraud_db'
   tab_name = 'events'
   mongo_cols = {'acct_type','user_type','email_domain','venue_state','venue_name'}
   client = MongoClient()
   db = client[client_name]
   tab = db[tab_name]
   cursor = tab.find(None)        #,mongo_cols)
   df = pd.DataFrame(list(cursor))
   return df




if name == __main__():	
	# read data via API client
	api = EventAPIClient()
	dataframe = api.get_data()
	# Read data from mongodb
	#dataframe = get_data()
	# clean data
	clean = Cleaner()
	X = clean.transform(dataframe)	
	# load pickled GB Model
	with open('data/model.pkl', 'rb') as f:
    	model = pickle.load(f)
    # Make Predictions
    prediction = model.predict(X.values)
    probability = round(np.max(Model.predict_proba(X))*100,2)
    print('''This posting is classified as {} with a probability 
            of {}%.'''.format(prediction, probability))
