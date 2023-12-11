# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:05:20 2023

@author: Mohamed Abdelnaby - Ph.D. Candidate at BU mabnaby@bu.edu
 Version 1.0
'''
"""
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

def calculate_moving_average(data, window_size):
    return data['Close'].rolling(window=window_size).mean()

def plot_moving_average(stock_data, ticker_symbol, window_size):
    # Create a copy of the data to avoid modifying the original DataFrame
    data = stock_data.copy()

    # Calculate the moving average
    data['Moving Average'] = calculate_moving_average(data, window_size)

    # Plot the closing prices and the moving average
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['Moving Average'], label=f'Moving Average ({window_size} days)')
    plt.title(f"{ticker_symbol} Stock Price and Moving Average")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Specify the path to the folder containing stock data
    stocks_folder = "./stocks"

    # Get a list of all stock files in the folder
    stock_files = [f for f in os.listdir(stocks_folder) if f.endswith('.csv')]

    # Choose a random stock file
    random_stock_file = random.choice(stock_files)

    # Load the randomly chosen stock data into a DataFrame
    stock_data = pd.read_csv(os.path.join(stocks_folder, random_stock_file))

    # Choose a window size for the moving average
    window_size = 20

    # Extract the ticker symbol from the file name
    ticker_symbol = os.path.splitext(random_stock_file)[0]

    # Plot the closing prices and moving average for the randomly chosen stock
    plot_moving_average(stock_data, ticker_symbol, window_size)
