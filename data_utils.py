import pandas as pd
import numpy as np
import os.path
from matplotlib import pyplot as plt

# Data paths
RAW_DATA = "data/Interview_data.csv"
GROUND_TRUTH = "data/ground_truth.csv"
TRAINING_DATA = "data/training.csv"
MISSING = "data/missing.csv"
PREDICTIONS = "data/predictions.csv"


def get_training_data():
    """ Get the cleaned and engineered training data
    """
    print("Getting Training Data")
    df = clean_data(pd.read_csv(RAW_DATA))
    df = engineer_features(df)
    df = create_training_data(df)
    return df.to_numpy()

def get_missing():
    """ Get rows with missing payments
    """
    raw = pd.read_csv(RAW_DATA)
    df = clean_data(raw.copy())
    df = engineer_features(df)
    return engineer_missing(raw, df).to_numpy()

def clean_data(df):
    """ Do some general data cleaning
    """

    print("Cleaning data...", end="   ")

    # Get rid of extra columns
    df.drop(df.columns[3:len(df.columns)], axis=1, inplace=True)
    
    # There are some empty cells in the PayDates.
    df['PayDate'] = df['PayDate'].ffill()
    df['PayDate'] = pd.arrays.DatetimeArray(pd.to_datetime(df['PayDate'], utc=True))
    
    # There are some null values in the Paydates
    df.dropna(subset=['PayAmt'], inplace=True)
    df.drop(df.loc[df['PayAmt']=="Null"].index, inplace=True)
    df.dropna(subset=['PatNum'], inplace=True)

    print("Done")
    return df

def engineer_features(df):
    """ Engineer several more features from our data
    """

    print("Engineering features...", end="   ")

    # Access from file so we don't have to engineer data each time
    if(os.path.exists(GROUND_TRUTH)):
        return pd.read_csv(GROUND_TRUTH)
    
    # Add an elapsed days feature
    df['StartDate'] = pd.to_datetime(df['PayDate'].loc[0])
    df['ElapsedDays'] = (df['PayDate'] - df['StartDate']).dt.days.astype(int)
    
    # Add a sum of all payments feature
    agg, num_payments = agg_payments(df)
    df['AvgPay'] = agg
    
    # Add a number of payments feature
    df['NumPays'] = num_payments
    
    # Add day, month, year, and weekday features
    df['Day'] = df['PayDate'].dt.day
    df['Month'] = df['PayDate'].dt.month
    df['Year'] = df['PayDate'].dt.year
    df['Weekday'] = df['PayDate'].dt.weekday
    
    # Sace the ground truth data
    df.to_csv(GROUND_TRUTH, index=False)
    print("Done")
    return df

def create_training_data(df):
    """ Create data for our neural network to train on
    """
    # Access file if data exists
    if(os.path.exists('data/training.csv')):
        return pd.read_csv(TRAINING_DATA)
    
    # Drop unnecessary columns
    df.drop('PayDate', inplace=True, axis=1)
    df.drop('StartDate', inplace=True, axis=1)
    df.drop('PatNum', inplace=True, axis=1)

    # Save trainng data to file
    df.to_csv(TRAINING_DATA, index=False)
    return df

def engineer_missing(raw, df):
    """ Get rows that are missing payment data. Cannot be passed the ground truth.
    """

    if(os.path.exists(MISSING)):
        return pd.read_csv(MISSING)

    # Cleaning
    raw.drop(raw.columns[3:len(raw.columns)], axis=1, inplace=True)
    raw['PayDate'] = raw['PayDate'].ffill()
    raw['PayDate'] = pd.arrays.DatetimeArray(pd.to_datetime(raw['PayDate'], utc=True))

    raw['StartDate'] = pd.to_datetime(raw['PayDate'].loc[0])
    raw['ElapsedDays'] = (raw['PayDate'] - raw['StartDate']).dt.days.astype(int)

    raw.drop(df.loc[df['PayAmt']=="Null"].index, inplace=True)
    raw.dropna(subset=['PatNum'], inplace=True)

    raw = raw.loc[pd.isna(raw['PayAmt'])]
    
    raw['ElapsedDays'] = df['ElapsedDays']
    

    avg_pay = []
    num_pay = []
    for i, row in raw.iterrows():
        user_data = df.loc[df['PatNum'] == raw['PatNum'].loc[i]]
        if len(user_data) > 0:
            avg_pay.append(user_data['AvgPay'].iloc[0])
            num_pay.append(user_data['NumPays'].iloc[0])

    raw['AvgPay'] = avg_pay
    raw['NumPays'] = num_pay

    raw.drop('PayAmt', inplace=True, axis=1)

    raw['Day'] = raw['PayDate'].dt.day
    raw['Month'] = raw['PayDate'].dt.month
    raw['Year'] = raw['PayDate'].dt.year
    raw['Weekday'] = raw['PayDate'].dt.weekday

    raw.drop('PayDate', inplace=True, axis=1)
    raw.drop('PatNum', inplace=True, axis=1)
    raw = raw.iloc[:,1:]

    raw.to_csv(MISSING, index=False)
    return raw.to_numpy()


def split_data(arr):
    """ Split data into test and train data. Accepts a numpy array as argument, not a dataframe
    """
    sep = int(np.floor(0.9 * len(arr[0])))
    return arr[:sep,1:], arr[:sep,0], arr[sep:,1:], arr[sep:,0]

def agg_payments(df):
    """ Compute the sum of each user's payments
    """
    print("   Aggragating payments...", end="   ")
    agg = []
    num_payments = []
    for i, row in df.iterrows():
        pat_rows = df.loc[df['PatNum'] == i]
        num_payments.append(len(pat_rows))
        agg.append(pat_rows['PayAmt'].astype(float).mean())

    print("Done")
    return agg, num_payments

def get_day_pay_data(df):
    """Get payments based on days of the week
    """
    
    data = [[],[]]
    for i in range(int(df['ElapsedDays'].iloc[-1])):
        day_data = df.loc[df['ElapsedDays'] == i]
        if(len(day_data) > 0):
            data[0].append(day_data['Weekday'].iloc[0])
            data[1].append(day_data['PayAmt'].astype(float).sum())
    return np.array(data)

def plot_date_price(df):
    """ Plot graph of payments based on time
    """
    plt.scatter(x =df['ElapsedDays'].to_list(), y = df['PayAmt'].astype(float).to_list())
    # if(os.path.exists(PREDICTIONS)):
    #     preds = pd.read_csv(PREDICTIONS)
    #     print(preds)
    #     plt.scatter(preds.iloc[0,:], preds.iloc[1,:])
    plt.xlabel("Elapsed Time (Days)")
    plt.ylabel("Amount Payed (Dollars)")
    plt.xlim(0,8312)
    plt.ylim(0,10000)
    plt.show()

def plot_pat_payments(df, pat_num):
    """ Plot graph of the payments of an individual patient
    """
    pat_data = df.loc[df['PatNum'] == pat_num]
    plt.scatter(x=pat_data['ElapsedDays'], y = pat_data['PayAmt'].astype(float))
    if(os.path.exists(PREDICTIONS)):
        preds = pd.read_csv(PREDICTIONS)
        plt.scatter(preds.iloc[0,2], preds.iloc[1,2])
    plt.xlim(0,1000)
    plt.ylim(0,1000)
    plt.xlabel("Elapsed Time (Days)")
    plt.ylabel("Amount Payed (Dollars)")
    plt.show()

def get_freq_price_data(df):
    """ Get data based on frequency and prices for each user
    """
    df = df.copy()
    prices = []
    freqs = []

    while len(df) > 0:
        print(len(df))
        user_data = df.loc[df['PatNum'] == df['PatNum'].iloc[0]]
        freq = (user_data['ElapsedDays'].max() - user_data['ElapsedDays'].min()) / len(user_data['ElapsedDays'])
        avg_price = user_data['PayAmt'].astype(float).mean()
        prices.append(avg_price)
        freqs.append(freq)
        df = df[df['PatNum'] != df['PatNum'].iloc[0]]
    return freqs, prices


if(__name__ == "__main__"):
    df = pd.read_csv(GROUND_TRUTH)
    plot_date_price(df)
    plot_pat_payments(df, 6)
    
    