import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import data_utils


def plot_date_price(df):
    print(df['PayAmt'].astype(float).to_list())
    plt.scatter(x =df['ElapsedDays'].to_list(), y = df['PayAmt'].astype(float).to_list())
    plt.xlim(0,1000)
    plt.ylim(0.10000)
    plt.show()


if(__name__ == "__main__"):
    df = data_utils.clean_data(pd.read_csv('data/Interview_data.csv'))
    plot_date_price(df)