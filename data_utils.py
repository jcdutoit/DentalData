import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def clean_data(df):
    # Get rid of extra columns
    df.drop(df.columns[3:len(df.columns)], axis=1, inplace=True)
    
    # There are some empty cells in the PayDates.
    df['StartDate'] = pd.to_datetime(df['PayDate'].loc[0])
    df['PayDate'] = df['PayDate'].ffill()
    df['PayDate'] = pd.arrays.DatetimeArray(pd.to_datetime(df['PayDate'], utc=True))
    
    df['ElapsedDays'] = (df['PayDate'] - df['StartDate']).dt.days.astype(int)
    df.dropna(subset=['PayAmt'], inplace=True)
    df.drop(df.loc[df['PayAmt']=="Null"].index, inplace=True)

    return df

def get_data_for_user(df, pat_num):
    return df.loc[df['PatNum'] == pat_num]

def plot_date_price(df):
    plt.scatter(x =df['ElapsedDays'].to_list(), y = df['PayAmt'].astype(float).to_list())        
    plt.xlim(0,1000)
    plt.ylim(0,2000)
    plt.show()

def plot_pat_payments(df, pat_num):

    pat_data = get_data_for_user(df,pat_num)
    plt.scatter(x=pat_data['ElapsedDays'], y = pat_data['PayAmt'].astype(float))
    plt.xlim(0,1000)
    plt.ylim(0,2000)
    plt.show()

def split_data(df):
    sep = np.floor(df.)
    return df.loc[:,:sep], df.loc[:,sep:]

if(__name__ == "__main__"):
    df = clean_data(pd.read_csv('data/Interview_data.csv'))
    plot_date_price(df)
    
    for i in range(400):
        plot_pat_payments(df, i)