# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:36:20 2023

@author: Mohamed Abdelnaby - Ph.D. Candidate at BU mabnaby@bu.edu
 Version 1.0
'''
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import random


# Set the path to your data folders
stocks_folder = "./stocks"
etfs_folder = "./etfs"
metadata_path = "./symbols_valid_meta.csv"

# Function to load data for a specific ticker
def load_ticker_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Date'])

# Function to load metadata
def load_metadata(file_path):
    return pd.read_csv(file_path)

# Function to make a dict of all stocks
def load_all_ticker_data(folder_path):
    stock_data_dict = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            ticker_symbol = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            stock_data_dict[ticker_symbol] = load_ticker_data(file_path)

    return stock_data_dict

# Function to visualize a subset of stocks
def visualize_random_subset(stock_data_dict, metadata=None, subset_size=10, use_plotly=False):
    # Randomly sample a subset of tickers
    all_tickers = list(stock_data_dict.keys())
    sampled_tickers = random.sample(all_tickers, min(subset_size, len(all_tickers)))

    # Create a dictionary with the sampled stock data
    sampled_stock_data = {ticker: stock_data_dict[ticker] for ticker in sampled_tickers}
        
    if use_plotly:
        # Interactive Plotly plot
        fig = px.line()
        for ticker_symbol, data in sampled_stock_data.items():
            fig.add_scatter(x=data.index, y=data['Close'], mode='lines', name=f"{ticker_symbol} - Close Price")

        fig.update_layout(
            title="Closing Prices Over Time",
            xaxis_title="Date",
            yaxis_title="Close Price",
            legend_title="Stocks",
        )

        if metadata is not None:
            fig.update_layout(legend=dict(x=1, y=1))
        
        fig.show()

    else:
        # Matplotlib plot
        # Find the common start date as the minimum of the minimum dates for all stocks
        min_start_date = min(min(stock_data['Date']) for ticker, stock_data in stock_data_dict.items())

        for ticker, data in sampled_stock_data.items():
            data['DaysSinceStart'] = (data['Date'] - min_start_date).dt.days
        for ticker_symbol, data in sampled_stock_data.items():
            plt.plot(data['DaysSinceStart'], data['Close'], label=f"{ticker_symbol} - Close Price")
          
        # Calculate mean values for each date across all stocks
        all_close_values = pd.concat([data.set_index('Date')['Close'] for data in stock_data_dict.values()], axis=1)

        # Replace NaN values with zero
        all_close_values = all_close_values.fillna(0)

        # Calculate mean values, considering only non-zero entries
        non_zero_count = all_close_values.astype(bool).sum(axis=1)
        mean_values = all_close_values.sum(axis=1) / non_zero_count
        
        # Plot the mean value
        plt.plot(list(range(len(mean_values))), mean_values, label='Mean Close Price', linestyle='--', color='black')
       
        plt.title("Closing Prices Over Time")
        plt.xlabel("Date")
        plt.ylabel("Close Price")

        if metadata is not None:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        else:
            plt.legend()

        plt.show()

metadata = load_metadata(metadata_path)
stock_data_dict = load_all_ticker_data(stocks_folder)
visualize_random_subset(stock_data_dict, metadata, subset_size=10, use_plotly=False)


# Function to visualize time series data
# def visualize_time_series(data, ticker_symbol, metadata=None):
#     plt.plot(data.index, data['Close'], label='Close Price')
#     plt.title(f"{ticker_symbol} - Closing Prices Over Time")
#     plt.xlabel("Date")
#     plt.ylabel("Close Price")
    
#     if metadata is not None:
#         full_name = metadata.loc[metadata['Symbol'] == ticker_symbol, 'Security Name'].values[0]
#         exchange = metadata.loc[metadata['Symbol'] == ticker_symbol, 'Listing Exchange'].values[0]
#         plt.suptitle(f"({ticker_symbol}) {full_name} - Exchange: {exchange}")

#     plt.legend()
#     plt.show()
    

# Plotting 5884 stocks for 10 years is not feasible and plotting them all together gets messy
# Iterate through the 'stocks' folder
# for filename in os.listdir(stocks_folder):
#     if filename.endswith(".csv"):
#         ticker_symbol = os.path.splitext(filename)[0]
#         file_path = os.path.join(stocks_folder, filename)
#         stock_data = load_ticker_data(file_path)

#         # Load metadata
#         metadata = load_metadata(metadata_path)

#         # Visualize time series data
#         plt.figure(figsize=(12, 6))
#         visualize_time_series(stock_data, ticker_symbol, metadata)

