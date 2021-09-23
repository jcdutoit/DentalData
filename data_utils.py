import pandas as pd
import numpy as np
import os.path
from matplotlib import pyplot as plt

def clean_data(df):
    # Get rid of extra columns
    df.drop(df.columns[3:len(df.columns)], axis=1, inplace=True)
    
    # There are some empty cells in the PayDates.
    
    df['PayDate'] = df['PayDate'].ffill()
    df['PayDate'] = pd.arrays.DatetimeArray(pd.to_datetime(df['PayDate'], utc=True))
    
    df.dropna(subset=['PayAmt'], inplace=True)
    df.drop(df.loc[df['PayAmt']=="Null"].index, inplace=True)
    df.dropna(subset=['PatNum'], inplace=True)

    return df

def get_missing_payments(df):
    df.drop(df.columns[3:len(df.columns)], axis=1, inplace=True)
    df['PayDate'] = df['PayDate'].ffill()
    return df.loc[pd.isna(df['PayAmt'])]

def split_data(arr):
    sep = int(np.floor(0.9 * len(arr)))
    # arr = np.delete(arr, 1, axis=1)
    print(arr.shape)
    arr = np.delete(arr, 2, axis=1)
    x = np.delete(arr, 0, axis=1)
    print(x.shape)
    y = arr[:,1]
    return x[:sep,:], y[:sep], x[sep:,:], y[sep:]

def engineer_features(df):
    if(os.path.exists('training.csv')):
        return pd.read_csv('training.csv')

    agg, num_payments = agg_payments(df)
    df['AvgPay'] = agg
    df['NumPays'] = num_payments
    df['Day'] = df['PayDate'].dt.day
    df['Month'] = df['PayDate'].dt.month
    df['Year'] = df['PayDate'].dt.year
    df['Weekday'] = df['PayDate'].dt.weekday
    df.drop('PayDate', inplace=True, axis=1)
    df.to_csv('training.csv')
    return df

def agg_payments(df):
    print("Calculating aggregates...")
    agg = []
    num_payments = []
    for i, row in df.iterrows():
        pat_rows = df.loc[df['PatNum'] == df['PatNum'][i]]
        num_payments.append(len(pat_rows))
        agg.append(pat_rows['PayAmt'].astype(float).mean())
    return agg, num_payments
        
def get_data():
    df = engineer_features(df)

def plot_date_price(df):
    df['StartDate'] = pd.to_datetime(df['PayDate'].loc[0])
    df['ElapsedDays'] = (df['PayDate'] - df['StartDate']).dt.days.astype(int)
    plt.scatter(x =df['ElapsedDays'].to_list(), y = df['PayAmt'].astype(float).to_list())        
    plt.xlim(0,1000)
    plt.ylim(0,2000)
    plt.show()

def plot_pat_payments(df, pat_num):
    df['StartDate'] = pd.to_datetime(df['PayDate'].loc[0])
    df['ElapsedDays'] = (df['PayDate'] - df['StartDate']).dt.days.astype(int)
    pat_data = df.loc[df['PatNum'] == pat_num]
    plt.scatter(x=pat_data['ElapsedDays'], y = pat_data['PayAmt'].astype(float))
    plt.xlim(0,1000)
    plt.ylim(0,2000)
    plt.show()

if(__name__ == "__main__"):
    raw = pd.read_csv('data/Interview_data.csv')
    # df = clean_data(raw.copy())
    # plot_date_price(df.copy())
    
    # # for i in range(400):
    # #     plot_pat_payments(df, i)
    # engineer_features(df.copy())
    print(get_missing_payments(raw))