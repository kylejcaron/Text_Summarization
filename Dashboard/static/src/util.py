import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def barplotter(df, ax, period='Q'):
    subdf = df[['time','predicted_sentiment']]
    subdf['y1'] = subdf['predicted_sentiment'].replace({-1:0})
    subdf['y2'] = subdf['predicted_sentiment'].replace({1:0,-1:1})
    subdf.time = pd.to_datetime(subdf.time, unit='s')
    if period=='y':
        subdf['year'] = pd.to_datetime(subdf.time).dt.year
        gtable = subdf.groupby('year').sum()
        n = len(gtable.index)
        X = gtable.index
        Y1 = gtable.y1.values
        Y2 = gtable.y2.values
        if gtable.y1.max() > gtable.y2.max():
            max_val = gtable.y1.max()+1
        else:
            max_val = gtable.y2.max()+1
        ax.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
        ax.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
        return ax
    if period=='Q':
        subdf['year_quarter'] = pd.PeriodIndex(subdf.time, freq='Q')
        gtable = subdf.groupby(['year_quarter']).sum()
        n = len(gtable.index)
        X = gtable.index

        if gtable.y1.max() > gtable.y2.max():
            max_val = gtable.y1.max()+1
        else:
            max_val = gtable.y2.max()+1

        Y1 = gtable.y1.values
        Y2 = gtable.y2.values*-1
        sns.barplot(x=gtable.index, y=gtable.y1, color="#1789d8", edgecolor='white', ax=ax)
        sns.barplot(x=gtable.index, y=-gtable.y2, color="#ef3421", edgecolor='white', ax=ax)
        ax.set_xticklabels(gtable.index,rotation=45, fontsize=6)
        ax.set_ylim(-max_val,max_val)

        #plt.ylabel('Review Count')
        return ax
    else:
      raise ValueError

def lineplotter(df, ax):
    df.time = pd.to_datetime(df.time,unit='s')
    df.set_index('time', inplace=True)
    df['roll'] = df.predicted_sentiment.cumsum()
    sns.lineplot(range(len(df)), df.roll,ax=ax)
    ax.set_xticklabels(df.index.date[-50:],rotation=45, fontsize=6)
    ax.set_ylim(-max_val,max_val)
    return ax