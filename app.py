import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from PIL import Image

st.write("""
### stock app
predicts long entrypoints based on neural network classification
""")


#ticker = 'AAPL'


ticker = st.text_input('Input ticker symbol', 'BTC-USD')
ticker = str(ticker)

start = datetime.datetime(2010, 1, 1)
#end_time = datetime.datetime(2019, 1, 20)
end = datetime.datetime.now().date().isoformat()         # today



ticker

tickerData = yf.Ticker(ticker)
df = yf.download(tickers=ticker, start=start, end=end)


st.write(df)
st.line_chart(df.Close)



st.write("""
### Pytorch neural network classifications/predictions
""")

def plot_OHLC(data, ticker):

    #fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(3,1,1)



    #plt.figure(figsize=(15,5))
    #plt.title('{} price data to {} timeframe'.format(ticker))
    ax.plot(data['Open'],linewidth=0.1)
    ax.plot(data['High'],linewidth=0.1)
    ax.plot(data['Low'],linewidth=0.1)
    ax.plot(data['Close'],linewidth=0.1)
    ax.plot(data['Adj Close'],linewidth=0.1)
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    st.write(fig)


##################plot_OHLC(df, ticker)



# ------------------ START OF              -----------------------
# ------------------ MACHINE LEARNING PART -----------------------
import talib as ta
import joblib
import pandas as pd
#import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

#newest yahoo API
import yfinance as yahoo_finance

import talib as ta
import numpy as np

import matplotlib.pyplot as plt



def get_data(ticker):
    df = yf.download(tickers=ticker, start=start, end=end)
    df = df.reset_index()
    return df


def compute_technical_indicators(df):
    df['EMA5'] = ta.EMA(df['Adj Close'].values, timeperiod=5)
    df['EMA10'] = ta.EMA(df['Adj Close'].values, timeperiod=10)
    df['EMA15'] = ta.EMA(df['Adj Close'].values, timeperiod=15)
    df['EMA20'] = ta.EMA(df['Adj Close'].values, timeperiod=10)
    df['EMA30'] = ta.EMA(df['Adj Close'].values, timeperiod=30)
    df['EMA40'] = ta.EMA(df['Adj Close'].values, timeperiod=40)
    df['EMA50'] = ta.EMA(df['Adj Close'].values, timeperiod=50)

    df['EMA60'] = ta.EMA(df['Adj Close'].values, timeperiod=60)
    df['EMA70'] = ta.EMA(df['Adj Close'].values, timeperiod=70)
    df['EMA80'] = ta.EMA(df['Adj Close'].values, timeperiod=80)
    df['EMA90'] = ta.EMA(df['Adj Close'].values, timeperiod=90)

    df['EMA100'] = ta.EMA(df['Adj Close'].values, timeperiod=100)
    df['EMA150'] = ta.EMA(df['Adj Close'].values, timeperiod=150)
    df['EMA200'] = ta.EMA(df['Adj Close'].values, timeperiod=200)

    df['upperBB'], df['middleBB'], df['lowerBB'] = ta.BBANDS(df['Adj Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    df['SAR'] = ta.SAR(df['High'].values, df['Low'].values, acceleration=0.02, maximum=0.2)

    # we will normalize RSI
    df['RSI'] = ta.RSI(df['Adj Close'].values, timeperiod=14)

    df['normRSI'] = ta.RSI(df['Adj Close'].values, timeperiod=14) / 100.0

    df.tail()

    return df


def compute_features(df):
    # computes features for forest decisions
    df['aboveEMA5'] = np.where(df['Adj Close'] > df['EMA5'], 1, 0)
    df['aboveEMA10'] = np.where(df['Adj Close'] > df['EMA10'], 1, 0)
    df['aboveEMA15'] = np.where(df['Adj Close'] > df['EMA15'], 1, 0)
    df['aboveEMA20'] = np.where(df['Adj Close'] > df['EMA20'], 1, 0)
    df['aboveEMA30'] = np.where(df['Adj Close'] > df['EMA30'], 1, 0)
    df['aboveEMA40'] = np.where(df['Adj Close'] > df['EMA40'], 1, 0)

    df['aboveEMA50'] = np.where(df['Adj Close'] > df['EMA50'], 1, 0)
    df['aboveEMA60'] = np.where(df['Adj Close'] > df['EMA60'], 1, 0)
    df['aboveEMA70'] = np.where(df['Adj Close'] > df['EMA70'], 1, 0)
    df['aboveEMA80'] = np.where(df['Adj Close'] > df['EMA80'], 1, 0)
    df['aboveEMA90'] = np.where(df['Adj Close'] > df['EMA90'], 1, 0)

    df['aboveEMA100'] = np.where(df['Adj Close'] > df['EMA100'], 1, 0)
    df['aboveEMA150'] = np.where(df['Adj Close'] > df['EMA150'], 1, 0)
    df['aboveEMA200'] = np.where(df['Adj Close'] > df['EMA200'], 1, 0)

    df['aboveUpperBB'] = np.where(df['Adj Close'] > df['upperBB'], 1, 0)
    df['belowLowerBB'] = np.where(df['Adj Close'] < df['lowerBB'], 1, 0)

    df['aboveSAR'] = np.where(df['Adj Close'] > df['SAR'], 1, 0)

    df['oversoldRSI'] = np.where(df['RSI'] < 30, 1, 0)
    df['overboughtRSI'] = np.where(df['RSI'] > 70, 1, 0)


    # very important - cleanup NaN values, otherwise prediction does not work
    df=df.fillna(0).copy()

    df.tail()

    return df


def plot_train_data(df):
    # plot price
    plt.figure(figsize=(15,2.5))
    plt.title('Stock data ' + str(ticker))
    plt.plot(df['Date'], df['Adj Close'])
    #plt.title('Price chart (Adj Close) ' + str(ticker))
    plt.show()
    return None


def define_target_condition(df):

    # price higher later - bad predictive results
    #df['target_cls'] = np.where(df['Adj Close'].shift(-34) > df['Adj Close'], 1, 0)

    # price above trend multiple days later
    df['target_cls'] = np.where(df['Adj Close'].shift(-34) > df.EMA150.shift(-34), 1, 0)

    # important, remove NaN values
    df=df.fillna(0).copy()

    df.tail()

    return df


import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim,100)
        self.layer2 = nn.Linear(100, 30)
        self.layer3 = nn.Linear(30, 2)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.drop(x)
        x = F.relu(self.layer2(x))
        x = self.drop(x)
        x = F.softmax(self.layer3(x)) # To check with the loss function
        return x


#ticker='BP'
#ticker='ABBV'
#ticker='GILD'
#ticker='NGG'
#ticker='BPY'
#ticker='AIR'

#ticker = 'BIDU'
#ticker = 'AMZN'
#ticker = 'IBM'

#start_time = datetime.datetime(1980, 1, 1)
#end_time = datetime.datetime(2019, 1, 20)
#end_time = datetime.datetime.now().date().isoformat()         # today


def plot_stock_prediction(df, ticker):
    # plot  values and significant levels
    #fig = plt.figure(figsize=(30,7))
    fig = plt.figure(figsize=(10,7))
    ####ax = fig.add_subplot(1,2,1)

    #ax.title('Predictive model ' + str(ticker))
    plt.plot(df['Date'], df['Adj Close'], label='Adj Close', alpha=0.2)

    plt.plot(df['Date'], df['EMA10'], label='EMA10', alpha=0.2)
    plt.plot(df['Date'], df['EMA20'], label='EMA20', alpha=0.2)
    plt.plot(df['Date'], df['EMA30'], label='EMA30', alpha=0.2)
    plt.plot(df['Date'], df['EMA40'], label='EMA40', alpha=0.2)
    plt.plot(df['Date'], df['EMA50'], label='EMA50', alpha=0.2)
    plt.plot(df['Date'], df['EMA100'], label='EMA100', alpha=0.2)
    plt.plot(df['Date'], df['EMA150'], label='EMA150', alpha=0.99)
    plt.plot(df['Date'], df['EMA200'], label='EMA200', alpha=0.2)


    plt.scatter(df['Date'], df['Buy']*df['Adj Close'], label='Buy', marker='^', color='magenta', alpha=0.15)
    #ax.scatter(df.index, df['sell_sig'], label='Sell', marker='v')
    plt.legend()

    plt.savefig('prediction.png')

    #####st.write(fig)
    image = Image.open('prediction.png')
    st.image(image)

    return None

new_df = get_data(ticker)

new_df = compute_technical_indicators(new_df)

new_df = compute_features(new_df)

new_df=define_target_condition(new_df)


saved_model = torch.load("iris-pytorch.pkl")


def predict_timeseries(df):

    # making sure we have good dimensions
    # column will be rewritten later
    df['Buy'] = df['target_cls']

    for i in range(len(df)):
        X_cls_valid = [[df['aboveSAR'][i],df['aboveUpperBB'][i],df['belowLowerBB'][i],
                        df['normRSI'][i],df['oversoldRSI'][i],df['overboughtRSI'][i],
                        df['aboveEMA5'][i],df['aboveEMA10'][i],df['aboveEMA15'][i],df['aboveEMA20'][i],
                        df['aboveEMA30'][i],df['aboveEMA40'][i],df['aboveEMA50'][i],
                        df['aboveEMA60'][i],df['aboveEMA70'][i],df['aboveEMA80'][i],df['aboveEMA90'][i],
                        df['aboveEMA100'][i]]]

        x_test = Variable(torch.Tensor(X_cls_valid).float())

        #####print('x_test',x_test)


        #####print('i',i)
        prediction = np.argmax(saved_model(x_test[0]).detach().numpy(), axis=0)
        #####print('prediction', prediction)


        df['Buy'][i] = prediction


    print(df.head())

    return df


new_df = predict_timeseries(new_df)

st.write(new_df)

##############plot_stock_prediction(new_df, ticker)

# zoom in on the data
temp_df = new_df[-500:]

plot_stock_prediction(temp_df, ticker)











# ------------------ END OF                -----------------------
# ------------------ MACHINE LEARNING PART -----------------------
