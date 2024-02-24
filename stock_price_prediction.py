import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

import plotly.graph_objects as go

from tensorflow.python.client import device_lib
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM




def check_gpu():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def load_data(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="5y")
    data.drop(columns=['Dividends', 'Stock Splits', 'Open', 'High', 'Low', 'Volume'], inplace=True)
    data.rename(columns={'Close': 'actual'}, inplace=True)
    return data

def prepare_data(data, future_days):
    future = data[-future_days:].copy()
    future.rename(columns={'actual': 'hidden_actual'}, inplace=True)
    data = data[:-future_days]
    return data, future

def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['actual'].dropna().values.reshape(-1,1))
    return scaled_data, scaler

def create_training_data(scaled_data):
    X_train, y_train = [], []
    for i in range(60, scaled_data.shape[0]):
        X_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train

def build_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    return model

def compile_and_fit_model(model, X_train, y_train):
    model.compile(loss='mean_squared_error', optimizer='adam')
    with tf.device('/device:GPU:0'):
        model.fit(X_train, y_train, epochs=1000, batch_size=1024*4, verbose=0)
    return model

def predict_prices(model, X_train, data, scaler):
    train_predict = model.predict(X_train, verbose=0)
    train_predict = scaler.inverse_transform(train_predict)
    data.loc[data.index[60:60+len(train_predict)], 'train_predict'] = train_predict.reshape(-1)
    return data


def create_test_data(data, future, prediction_days, scaler):
    temp_input = np.concatenate((data['actual'][-prediction_days:].values, future['hidden_actual'].values))
    temp_input = scaler.transform(temp_input.reshape(-1, 1))
    X_test = []
    for i in range(prediction_days, temp_input.shape[0]):
        X_test.append(temp_input[i-prediction_days:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test

def add_predicted_prices_to_future(model, X_test, future, scaler):
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    future['predicted'] = predicted_prices
    return future

def plot_data(data, future, symbol):
    merged_data = pd.concat([data, future], axis=1)
    merged_data = merged_data.tail(90)
    # calculate RSME
    rmse = np.sqrt(np.mean(np.square(merged_data['hidden_actual'].tail(30) - merged_data['predicted'].tail(30))))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged_data.index, y=merged_data.actual, mode='lines', name='Actual', line=dict(color='blue', width=2, dash='solid')))
    fig.add_trace(go.Scatter(x=merged_data.index, y=merged_data.hidden_actual, mode='lines', name='Hidden Actual', line=dict(color='blue', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=merged_data.index, y=merged_data['train_predict'], mode='lines', name='train data', line=dict(color='red', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=merged_data.index, y=merged_data['predicted'], mode='lines', name='predicted data', line=dict(color='green', width=2, dash='dot')))
    fig.update_layout(title=f'[{symbol}] Stock Price [RMSE: {rmse:.2f}]', yaxis_title='Price', template='plotly_dark')
    fig.update_xaxes(showgrid=False)
    #fig.show()
    # save the plot as a .png file 
    fig.write_image(f"/home/joe/tensorflow/charts/{symbol}.png", width=1920, height=1080, scale=1)
    

def predict_stock_future_prices(symbol):
    future_days = 30
    prediction_days = 60

    data = load_data(symbol)
    data, future = prepare_data(data, future_days)
    scaled_data, scaler = normalize_data(data)
    X_train, y_train = create_training_data(scaled_data)
    model = build_model(X_train)
    model = compile_and_fit_model(model, X_train, y_train)
    data = predict_prices(model, X_train, data, scaler)
    X_test = create_test_data(data, future, prediction_days, scaler)
    future = add_predicted_prices_to_future(model, X_test, future, scaler)

    

    plot_data(data, future, symbol)


def main():
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'INTC', 'CSCO', 'ADBE', 'PYPL', 'NFLX', 'CMCSA', 'PEP', 'COST', 'TMUS', 'AVGO', 'QCOM', 'TXN', 'INTU', 'AMD', 'MU', 'AMAT', 'LRCX']    
    for symbol in symbols:
        predict_stock_future_prices(symbol)

if __name__ == "__main__":
    main()
