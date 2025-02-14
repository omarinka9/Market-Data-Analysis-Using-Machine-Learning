# Market-Data-Analysis-Using-Machine-Learning
Market Data Analysis Using Machine Learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_data(filename):
    df = pd.read_csv(filename, parse_dates=True, index_col='Date')
    return df

def split_data(df, test_size=0.2):
    num_rows = len(df)
    train_size = int(num_rows * (1 - test_size))
    X_train = df.iloc[:train_size][['Open']]
    y_train = df.iloc[:train_size][['Close']]
    X_test = df.iloc[train_size:][['Open']]
    y_test = df.iloc[train_size:][['Close']]
