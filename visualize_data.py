import pandas as pd
from matplotlib import pyplot as plt
import data_utils


def plot_date_price(df):
    plt.scatter(x =df['ElapsedDays'].to_list(), y = df['PayAmt'].astype(float).to_list())        
    plt.xlim(0,1000)
    plt.ylim(0,2000)
    plt.show()

def plot_pat_payments(df, pat_num):
    pat_data = data_utils.get_data_for_user(df,pat_num)
    plt.scatter(x=pat_data['ElapsedDays'].to_list(), y = pat_data['PayAmt'].astype(float).to_list())
    plt.xlim(0,1000)
    plt.ylim(0,2000)
    plt.show()
    
if(__name__ == "__main__"):
    df = data_utils.clean_data(pd.read_csv('data/Interview_data.csv'))
    plot_date_price(df)


    for i in range(400):
        plot_pat_payments(df, i)