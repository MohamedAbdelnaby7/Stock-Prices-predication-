import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tools.eval_measures import aic
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from pmdarima import auto_arima
import tqdm

file_directory = "archive/stocks" # "sample_stocks"
file_names = [file for file in os.listdir(file_directory) if file.endswith(".csv")]
stock_data = {}

# for file_name in file_names:
#     try:
#         stock_name = os.path.splitext(file_name)[0]
#         file_path = os.path.join(file_directory, file_name)

#         # Load data for each stock
#         df = pd.read_csv(file_path)
#         df['Date'] = pd.to_datetime(df['Date'])
#         df.set_index('Date', inplace=True)

#         # Check if DataFrame is not empty
#         if not df.empty:
#             stock_data[stock_name] = df
#         else:
#             print(f"Warning: DataFrame for {stock_name} is empty. Continuing to the next file.")

#     except Exception as e:
#         print(f"Error processing {file_name}: {e}")
#     # stock_name = os.path.splitext(file_name)[0]  # Extract stock name
#     # file_path = os.path.join(file_directory, file_name)
    
#     # # Load data for each stock
#     # stock_data[stock_name] = pd.read_csv(file_path)
#     # stock_data[stock_name]['Date'] = pd.to_datetime(stock_data[stock_name]['Date'])
#     # stock_data[stock_name].set_index('Date', inplace=True)

# print(list(stock_data.keys()))
# # print(stock_data)

# Function that checks for stationarity and does differencing if it is not.
# Function returns stationary data and number of times of differencing.
def check_stationarity(data, diff_count=0):
    # Checking for infinite or missing values and handle them
    if np.any(np.isinf(data.values)) or np.any(np.isnan(data.values)):
        print("Data contains inf or nans. Handling them now...")
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

    result = adfuller(data.values)
    ADF_statistic = result[0]
    p_value = result[1]
    critical_value = result[4]['5%']

    print('ADF statistic: ', ADF_statistic)
    print('p-value: ', p_value)
    print('Critical Values: ', critical_value)

    if (p_value <= 0.05) & (critical_value > ADF_statistic):
        print("Data is stationary")
        print(" ")
        return data, diff_count
    else:
        print("Data is not stationary, differencing them now...")
        print(" ")
        differenced_data = data.diff().dropna()
        # differenced_data.columns = data.columns
        return check_stationarity(differenced_data, diff_count + 1)
    

# Function that determines best lag order for AR (p) and MA (q)
# Values for p and q are based on autocorrelation function (ACF) and
# partial autocorrelation function (PACF)
def pick_p_q_AIC(data):
    model = auto_arima(data, 
                       start_p=0, start_q=0,
                       max_p=12, max_q=12,
                       suppress_warnings=True)
    orders = model.order

    # print(orders) # (p, d, q)
    # print(model.summary())

    return orders[0], orders[2]


def pick_p_q_from_acf_pacf(data, max_lag):
    # Plotting ACF and PACF plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(data, lags=max_lag, ax=ax1)
    plot_pacf(data, lags=max_lag, ax=ax2)
    plt.show()

def ARIMA_MSE_and_write_to_csv(file_batch, stock_data, feature_vector, train_ratio, plot_mode=False):
    all_results = []
    for file_name in tqdm.tqdm(file_batch, desc="Running and evaluating ARIMA"):
        stock_name = os.path.splitext(file_name)[0]
        file_path = os.path.join(file_directory, file_name)

        try:
            # Loading data for each stock
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            # Check if DataFrame is not empty
            if not df.empty:
                stock_data[stock_name] = df
            else:
                print(f"Warning: DataFrame for {stock_name} is empty. Continuing to the next file.")
                continue

            # Isolating feature vector
            feature_series = df[feature_vector]

            # Check stationarity and difference dataset when necessary
            stationary_data, diffs_count = check_stationarity(feature_series)
            print('d: ', diffs_count)

            # Picking the ideal hyperparameters
            p, q = pick_p_q_AIC(stationary_data)

            # Splitting data into training and test sets
            training_ratio = train_ratio
            training_size = int(len(feature_series) * training_ratio)
            training_data, test_data = stationary_data[:training_size], stationary_data[training_size:]

            # Training ARIMA model and making predictions
            ARIMA_model = ARIMA(training_data, order=(p, diffs_count, q))
            fit_model = ARIMA_model.fit()
            pred = fit_model.forecast(steps=len(test_data))

            # Normalizing data before calculating MSE
            scaler = StandardScaler()
            test_data_norm = scaler.fit_transform(test_data.values.reshape(-1, 1))
            pred_norm = scaler.transform(pred.values.reshape(-1, 1))

            MSE_norm = mean_squared_error(test_data_norm, pred_norm)

            MAPE = mean_absolute_percentage_error(test_data, pred)
            print(f"MAPE for {stock_name}: {MAPE}")

            # Append results to the batch results
            results = {
                'Stock_Name': stock_name,
                'Feature_Vector': feature_vector,
                'MSE': MSE_norm,
                'MAPE': MAPE
            }
            all_results.append(results)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    return all_results

# Batch size
batch_size = 500

# Process files in batches
num_batches = len(file_names) // batch_size
for batch_num in range(num_batches + 1):
    start_idx = batch_num * batch_size
    end_idx = (batch_num + 1) * batch_size
    current_batch = file_names[start_idx:end_idx]

    # Process the current batch
    all_results_batch = ARIMA_MSE_and_write_to_csv(current_batch, stock_data, 'Close', 0.8, plot_mode=False)

    # Save batch results to a CSV file
    batch_file_name = f'arima_batch_{batch_num}.csv'
    batch_csv_header = ['Stock_Name', 'Feature_Vector', 'MSE', 'MAPE']
    with open(batch_file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=batch_csv_header)
        writer.writeheader()
        for result in all_results_batch:
            writer.writerow(result)

# Concatenate all batch results into one CSV file
final_file_name = 'arima_final_results.csv'
with open(final_file_name, 'w', newline='') as final_csvfile:
    writer = csv.DictWriter(final_csvfile, fieldnames=batch_csv_header)
    writer.writeheader()
    for batch_num in range(num_batches + 1):
        batch_file_name = f'arima_batch_{batch_num}.csv'
        with open(batch_file_name, 'r') as batch_csvfile:
            batch_reader = csv.DictReader(batch_csvfile)
            for row in batch_reader:
                writer.writerow(row)

# Clean up: Remove individual batch files
# for batch_num in range(num_batches + 1):
#     batch_file_name = f'arima_batch_{batch_num}.csv'
#     os.remove(batch_file_name)

print(f'Final results saved to {final_file_name}')

    
# ARIMA_MSE_and_write_to_csv(stock_data, 'Close', 0.8, plot_mode=False)