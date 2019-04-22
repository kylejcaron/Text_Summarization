from flask import Flask, render_template, redirect, request
from flask_bootstrap import Bootstrap
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
import sqlite3 as sql
import psycopg2 as pg2
from sqlalchemy import create_engine
from flask_table import Table, Col
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from time import time
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from static.src.model import Model
from matplotlib import pyplot as plt
import pickle
import numpy as np
import io
import seaborn as sns
import base64
from static.src.util import lineplotter, barplotter

POSTGRES = {
    'user': 'kylejcaron',
    'pw': 'PSQL',
    'db': 'company_reviews',
    'host': 'localhost',
    'port': '5432',
}

app = Flask(__name__)
Bootstrap(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://%(user)s:\
%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES

sql_id = 'postgresql://%(user)s:\
%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES

# loads query from sql database to pandas df
def load_sql_table(query):
    try:
        df = pd.read_sql(query, con=connector)
        return df
    except:
        return None

@app.route('/')
def start_redirect():
    return redirect("http://127.0.0.1:5000/dashboard", code=302)

@app.route('/dashboard', methods=['GET', 'POST'])
def start_dashboard():
    # Ask for all tables in your SQL Database
    # Request might look different for non MySQL
    # E.g. for SQL Server: sql_statement = "SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'"
    # TODO: Modify for your needs
    sql_statement = """SELECT table_name
        FROM   information_schema.tables
        WHERE  table_schema = 'public'
        ORDER  BY 1;
    """
    tables = load_sql_table(sql_statement)

    if request.method == 'POST':
        whichTable = request.form['whichTable']
        # Load the requested table from Database
        # TODO: Set your database query for the chosen table (e.g. modify db schema)
        SQL_table = 'SELECT text, predicted_summary, predicted_score, score, predicted_proba, productid FROM {} ORDER BY predicted_proba DESC LIMIT 30;'.format(whichTable)
        # Declare your table
        table = load_sql_table(SQL_table)
        #result = table.reset_index().to_html(index = False,index_names = False)
        result = True
        return render_template('dashboard.html', tables=tables, cmd=result, table=table, selectedTable=whichTable)

    else:
        result = False
        return render_template('dashboard.html', tables=tables, cmd=result, table=result, selectedTable='None')

@app.route('/dashboard/product/', methods=['GET', 'POST'])
def product_dashboard_main():
    # Ask for all tables in your SQL Database
    # Request might look different for non MySQL
    # E.g. for SQL Server: sql_statement = "SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'"
    # TODO: Modify for your needs

    sql_statement = """SELECT DISTINCT("productid") FROM entries;"""
    tables = load_sql_table(sql_statement)

    if request.method == 'POST':
        whichProduct = request.form['whichProduct']
        print(whichProduct)
        # Load the requested table from Database
        # TODO: Set your database query for the chosen table (e.g. modify db schema)
        print('sql table')
        SQL_table = 'SELECT text, predicted_summary, predicted_score, score, predicted_proba, productid FROM entries WHERE productid like \'{}\' ORDER BY predicted_proba DESC LIMIT 30;'.format(whichProduct)
        print(SQL_table)
        # Declare your table
        table = load_sql_table(SQL_table)
        print(table)
        #result = table.reset_index().to_html(index = False,index_names = False)
        result = True
        return render_template('product_dashboard.html', PID = str(whichProduct), tables=tables, cmd=result, table=table, selectedTable=whichProduct)

    else:
        result = False
        whichProduct = ''
        return render_template('product_dashboard.html', PID = str(whichProduct),tables=tables, cmd=result, table=result, selectedTable='None')


@app.route('/dashboard/product/<pid>', methods=['GET', 'POST'])
def product_dashboard(pid):
   
    #sql_statement = """SELECT DISTINCT(\"{}\") FROM entries;""".format(pid)
    sql_statement = """SELECT DISTINCT("productid") FROM entries;"""
    tables = load_sql_table(sql_statement)
    whichProduct = pid
    # Load the requested table from Database
    # TODO: Set your database query for the chosen table (e.g. modify db schema)
    print('sql table')
    SQL_table = 'SELECT text, predicted_summary, predicted_score, score, predicted_proba, productid FROM entries WHERE productid like \'{}\' ORDER BY predicted_proba DESC LIMIT 30;'.format(whichProduct)
    # Declare your table
    table = load_sql_table(SQL_table)
    result = True

    # For wordclouds
    wordcloud_query = 'SELECT text FROM negative_reviews WHERE productid like \'{}\';'.format(whichProduct)
    wordcloud_data = load_sql_table(wordcloud_query)
    # Load pickled model
    with open('static/data/model.pkl', 'rb') as f:
        Model = pickle.load(f)

    # Use tf-idf features for NMF.
    tfidf = Model.tfidf.transform(wordcloud_data.text)
    # Fit the NMF model
    n_components = 10
    n_top_words = 10
    try:
        nmf = NMF(n_components=n_components, random_state=1,
                  beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
                  l1_ratio=.5).fit(tfidf)
    except:
        nmf = NMF(n_components=len(wordcloud_data.text), random_state=1,
                  beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
                  l1_ratio=.5).fit(tfidf)
    tfidf_feature_names = Model.tfidf.get_feature_names()
    #print_top_words(nmf, tfidf_feature_names, n_top_words)
    top_words = add_top_words(nmf, tfidf_feature_names, n_top_words)


    cloud_stopwords = {'amazon', 'wanted', 'product', 'mouth', 'today', 'sent', 'got', 'don', 'company', 'producer'
                      'item', 'ordered', 'dop', 'received', 'costco', 'bag', 'bought', 'know', 'thought', 'facts',
                      'february', 'item', 'times', 'going', 'buy', 'food', 'way', 'following', 'apparently', 'products', 'caused',
                      'noticed', 'reviews', 'previously', 'uses', 'cause', 'wish', 'ok', 'okay', 'forums', 'want', 'literally', 'giving'}

    top_words_cleaned = [word for word in top_words if word in Model.tfidf.vocabulary_.keys()]
    top_words_hashed = [Model.tfidf.vocabulary_[word] for word in top_words_cleaned]
    # 0 class are negative, 1 class are positive. Want to focus in on negative class
    class_arr = np.argmax(np.exp(Model.model.feature_log_prob_)[:,top_words_hashed],axis=0)

    neg_word_list = []
    for idx,word in enumerate(top_words_cleaned):
        if class_arr[idx] == 0:
            if word not in cloud_stopwords:
                neg_word_list.append(word)


    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,4),constrained_layout=True)
    buf = io.BytesIO()
    wordcloud = WordCloud(background_color='white', colormap='tab10').generate(' '.join(neg_word_list))
    ax1.imshow(wordcloud, interpolation='bilinear')
    ax1.axis('off')
    #ax2
    # time_query = 'SELECT time, predicted_score FROM entries WHERE productid LIKE \'{}\' ORDER BY time ASC'.format(whichProduct)
    time_query = 'SELECT time, predicted_sentiment FROM entries WHERE productid LIKE \'{}\' ORDER BY time ASC'.format(whichProduct)

    timedf = load_sql_table(time_query)
    timedf = timedf.sort_values(by=['time'])


    barplotter(timedf,ax2)
    #ax2.set_ylabel('Count of Negative/Positive Reviews')
    # lineplotter(timedf,ax2)
    ax2.set_ylabel('Sentiment')
    ax2.set_xlabel('Year')
    #ax2.set_aspect(1)
    #ax2.figure.set_size_inches(w=6,h=4)
    plt.axis('on')
    #plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    buffer = b''.join(buf)
    b2 = base64.b64encode(buffer)
    image=b2.decode('utf-8')


    if request.method == 'POST':
        whichProduct = request.form['whichProduct']
        print(whichProduct)
        # Load the requested table from Database
        # TODO: Set your database query for the chosen table (e.g. modify db schema)
        print('sql table')
        SQL_table = 'SELECT text, predicted_summary, predicted_score, score, predicted_proba, productid FROM entries WHERE productid like \'{}\' ORDER BY predicted_proba DESC LIMIT 30;'.format(whichProduct)
        print(SQL_table)
        # Declare your table
        table = load_sql_table(SQL_table)
        print(table)
        #result = table.reset_index().to_html(index = False,index_names = False)
        result = True
        return render_template('product_dashboard_detail.html', PID = str(whichProduct), img=image, tables=tables, cmd=result, table=table, selectedTable=whichProduct)
    else:
        result = True
        return render_template('product_dashboard_detail.html', PID = str(whichProduct),tables=tables,  img=image, cmd=result, table=table, selectedTable=whichProduct)

@app.route('/dashboard/analytics/', methods=['GET', 'POST'])
def product_analytics():
    total_sentiment = 'SELECT productid, sum(predicted_sentiment), ROUND(sum(predicted_score)/count(predicted_sentiment),2)*100, count(predicted_sentiment) FROM entries GROUP BY productid ORDER BY sum(predicted_sentiment) ASC;'
    rolling_sentiment = 'SELECT time, productid, sum(predicted_sentiment) OVER (ORDER BY time) FROM entries GROUP BY time, predicted_sentiment, productid ORDER BY time;'
    total_sentiment_table = load_sql_table(total_sentiment)
    rolling_sentiment_table = load_sql_table(rolling_sentiment)
    rolling_sentiment_table.time = pd.to_datetime(rolling_sentiment_table.time,unit='s')    
    # tables = pd.DataFrame(['Recent Reviews','Sentiment Dashboard'])
    tables = pd.DataFrame(['Sentiment Dashboard'])


    if request.method == 'POST':
        whichTable = request.form['whichTable']
        if whichTable=='Recent Reviews':
            return render_template('analytics.html', tables=tables, table=rolling_sentiment_table, selectedTable=whichTable)
        else:
            whichTable = 'Sentiment Dashboard'
            return render_template('analytics.html', tables=tables, table=total_sentiment_table, selectedTable=whichTable)
    else:
        whichTable = 'Sentiment Dashboard'
        return render_template('analytics.html', tables=tables, table=total_sentiment_table, selectedTable=whichTable)
    
def add_top_words(model, feature_names, n_top_words):
    word_list = []
    for topic_idx, topic in enumerate(model.components_):
        words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        for word in words:
            word_list.append(word)
    return word_list

if __name__ == '__main__':
    # connect to your database
    try:
        # TODO: Use Library of your needs
        # E.g. for SQL Server it might be pyodbc
        # use 127.0.0.1 if localhost
        connector = pg2.connect(dbname = POSTGRES['db'], host = POSTGRES['host'],
          user = POSTGRES['user'], password = POSTGRES['pw'])
    except:
        print("No access to the required database")
    app.run()
