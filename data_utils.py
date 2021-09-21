import pandas as pd
import numpy as np

def clean_data(df):
    
    # Get rid of extra columns
    df.drop(df.columns[3:len(df.columns)], axis=1, inplace=True)
    
    # There are some empty cells in the PayDates.
    
    df['StartDate'] = pd.to_datetime(df['PayDate'].loc[0])
    df['PayDate'] = df['PayDate'].ffill()
    # df['PayDate'] = (df['PayDate'])
    df['PayDate'] = pd.arrays.DatetimeArray(pd.to_datetime(df['PayDate'], utc=True))
    
    df['ElapsedDays'] = (df['PayDate'] - df['StartDate']).dt.days.astype(int)
    df.dropna(subset=['PayAmt'], inplace=True)
    df.drop(df.loc[df['PayAmt']=="Null"].index, inplace=True)
    # df.to_csv('test.csv')

    return df



if(__name__ == "__main__"):
    df = pd.read_csv('./data/Interview_data.csv')
    df = clean_data(df)
    print(df.head())
    
