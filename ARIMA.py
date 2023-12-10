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

file_directory = "archive/stocks"# "sample_stocks"
file_names = [file for file in os.listdir(file_directory) if file.endswith(".csv")]
stock_data = {}

for file_name in file_names:
    stock_name = os.path.splitext(file_name)[0]  # Extract stock name
    file_path = os.path.join(file_directory, file_name)
    
    # Load data for each stock
    stock_data[stock_name] = pd.read_csv(file_path)
    stock_data[stock_name]['Date'] = pd.to_datetime(stock_data[stock_name]['Date'])
    stock_data[stock_name].set_index('Date', inplace=True)

print(list(stock_data.keys()))
# print(stock_data)

# Function that checks for stationarity and does differencing if it is not.
# Function returns stationary data and number of times of differencing.
def check_stationarity(data, diff_count=0):
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

# Function that determines orders for ARIMA using AIC/grid search
def find_best_arima_order(data, p_range=range(3), d_range=range(3), q_range=range(3)):
    best_aic = float('inf')
    best_order = None

    # Grid search
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(data, order=(p, d, q), enforce_invertibility=False)
                    result = model.fit()

                    # Calculating AIC
                    current_aic = aic(result.llf, len(result.params))

                    # Update best model if current AIC is lower
                    if current_aic < best_aic:
                        best_aic = current_aic
                        best_order = (p, d, q)

                except Exception as e:
                    # Model fails to converge
                    print(f"Exception: {e}")
                    continue

    return best_aic, best_order

def pick_p_q_from_acf_pacf(data, max_lag):
    # Plotting ACF and PACF plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(data, lags=max_lag, ax=ax1)
    plot_pacf(data, lags=max_lag, ax=ax2)
    plt.show()

all_results = []
def ARIMA_MSE_and_write_to_csv(stock_data, feature_vector, train_ratio, plot_mode=False):
    # Initializing data frame to store information
    # results_df = pd.DataFrame(columns=['Stock_Name', 'Feature_Vectors', 'MSE', 'Test_Data', 'Predicted_Data'])
    
    for stock_name, stock_df in tqdm.tqdm(stock_data.items(), desc="Running and evaluating ARIMA"):
        # Isolating feature vector
        feature_series = stock_df[feature_vector]

        # # Normalize feature vector
        # scaler = StandardScaler()
        # feature_series_norm = scaler.fit_transform(feature_series.values.reshape(-1,1))

        # Check stationarity and difference dataset when necessary
        stationary_data, diffs_count = check_stationarity(feature_series)
        # print('d: ', diffs_count)

        # Picking the ideal hyperparameters
        p, q = pick_p_q_AIC(stationary_data)
        # print("p: ", p)
        # print("q: ", q)

        # pick_p_q_from_acf_pacf(stationary_data, 100)
        # best_aic, best_order = find_best_arima_order(stationary_data)
        # print("Best AIC:", best_aic)
        # print("Best Order (p, d, q):", best_order)

        # Splitting data into training and test sets
        training_ratio = train_ratio
        training_size = int(len(feature_series) * training_ratio)
        training_data, test_data = stationary_data[:training_size], stationary_data[training_size:]

        # Training ARIMA model and making predictions
        ARIMA_model = ARIMA(training_data, order=(p, diffs_count, q))
        fit_model = ARIMA_model.fit()
        pred = fit_model.forecast(steps=len(test_data))

        # Evaluating performace of the model with MSE
        # MSE = mean_squared_error(test_data, pred)
        # # print('original test data: ', test_data)
        # # print('original prediction: ', pred)
        # print(f'MSE for original {stock_name}: {MSE}')

        # Normalizing data before calculating MSE
        scaler = StandardScaler()
        test_data_norm = scaler.fit_transform(test_data.values.reshape(-1,1))
        pred_norm = scaler.transform(pred.values.reshape(-1,1))

        MSE_norm = mean_squared_error(test_data_norm, pred_norm)
        # print('normalized test data: ', test_data)
        # print('normalized prediction: ', pred)
        # print(f'MSE for normalized {stock_name}: {MSE_norm}')

        # Evaluating model performance with MPE
        MAPE = mean_absolute_percentage_error(test_data, pred)
        print(f"MAPE for {stock_name}: {MAPE}")
        
        if plot_mode:
            plt.figure(figsize=(10,6))
            plt.plot(training_data, label='Training Set', color='Blue')
            plt.plot(test_data, label='Test Set', color='Gray')
            plt.plot(test_data.index, pred, label='ARIMA pred', color='Orange')
            plt.title(f'ARIMA model for {stock_name} - {feature_vector}')
            plt.xlabel('Date')
            plt.ylabel(f'{feature_vector}')
            plt.grid(True)
            plt.legend()
            plt.show()


        # write results to CSV
        results = {
            'Stock_Name' : stock_name,
            'Feature_Vector' : feature_vector,
            'MSE' : MSE_norm,
            'MAPE' : MAPE#,
            # 'Test_Data' : test_data.to_list(),
            # 'Predicted_Data' : pred.to_list()
        }
        all_results.append(results)
        
        tqdm.tqdm.write(f"Processing {stock_name}")

    file_name = f'arima_{feature_vector}.csv'
    csv_header = ['Stock_Name', 'Feature_Vector', 'MSE', 'MPE'] # , 'Test_Data', 'Predicted_Data']

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    


ARIMA_MSE_and_write_to_csv(stock_data, 'Close', 0.8, plot_mode=False)

# stationary_data, num_diffs = check_stationarity(catfish_sales['Total'])
# stationary_data, num_diffs = check_stationarity(subset_catfish_sales['Total'])
# print(stationary_data)