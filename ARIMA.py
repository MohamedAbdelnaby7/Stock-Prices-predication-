import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pmdarima import auto_arima
import tqdm

"""
Before running the code, make sure that the path stock files are specified under
file_directory. If it is at the same directory, simply need to only specify name of the
folder.
"""
file_directory = "sample_stocks" # "archive/stocks"
file_names = [file for file in os.listdir(file_directory) if file.endswith(".csv")]
stock_data = {}

# Gathering stock from files
for file_name in file_names:
    try:
        stock_name = os.path.splitext(file_name)[0]
        file_path = os.path.join(file_directory, file_name)

        # Load data for each stock
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Check if DataFrame is not empty
        if not df.empty:
            stock_data[stock_name] = df
        else:
            print(f"Warning: DataFrame for {stock_name} is empty. Continuing to the next file.")

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Function that checks for stationarity and does differencing if it is not.
# Function returns stationary data and number of times of differencing.
def check_stationarity(data, diff_count=0):
    # Checking for infinite or missing values and handle them
    if np.any(np.isinf(data.values)) or np.any(np.isnan(data.values)):
        print("Data contains inf or nans. Handling them now...")
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

    # Utilizing Augmented-Dickey Fuller test ti check for stationarity
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

        return check_stationarity(differenced_data, diff_count + 1)
    

# Function that determines best lag order for AR (p) and MA (q)
# with Akaike Information Criterion method
def pick_p_q_AIC(data):
    model = auto_arima(data, 
                       start_p=0, start_q=0,
                       max_p=12, max_q=12,
                       suppress_warnings=True)
    orders = model.order

    return orders[0], orders[2]


# Function that plots ACF and PACF plot
def plotting_acf_pacf(data, max_lag):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(data, lags=max_lag, ax=ax1)
    plot_pacf(data, lags=max_lag, ax=ax2)
    plt.show()


# Function that runs and evaluates ARIMA with MSE
all_results = []
def ARIMA_MSE_and_write_to_csv(stock_data, feature_vector, train_ratio, plot_mode=False):  
    for stock_name, stock_df in tqdm.tqdm(stock_data.items(), desc="Running and evaluating ARIMA"):
        # Isolating feature vector
        feature_series = stock_df[feature_vector]

        # Check stationarity and difference dataset when necessary
        stationary_data, diffs_count = check_stationarity(feature_series)
        # print('d: ', diffs_count)

        # Picking the ideal hyperparameters
        p, q = pick_p_q_AIC(stationary_data)
        # print("p: ", p)
        # print("q: ", q)

        # Splitting data into training and test sets
        training_ratio = train_ratio
        training_size = int(len(feature_series) * training_ratio)
        training_data, test_data = stationary_data[:training_size], stationary_data[training_size:]

        # Training ARIMA model and making predictions
        ARIMA_model = ARIMA(training_data, order=(p, diffs_count, q))
        fit_model = ARIMA_model.fit()
        pred = fit_model.predict(start=len(training_data), end=len(training_data) + len(test_data) - 1, dynamic=False, typ='levels')

        # Evaluating performace of the model with MSE
        MSE = mean_squared_error(test_data, pred)
        # # print('original test data: ', test_data)
        # # print('original prediction: ', pred)
        print(f'MSE for {stock_name}: {MSE}')

        # Evaluating model performance with MPE
        MAPE = mean_absolute_percentage_error(test_data, pred)
        print(f"MAPE for {stock_name}: {MAPE}")
        
        if plot_mode:
            plt.figure(figsize=(10,6))
            plt.plot(training_data, label="Training Set", color='Blue')
            plt.plot(test_data, label="Test Set", color='Gray')
            plt.plot(test_data.index, pred, label="Predictions", color='Orange')
            plt.title(f'ARIMA model for {stock_name} - {feature_vector} at train ratio={train_ratio}')
            plt.xlabel('Date')
            plt.ylabel(f'{feature_vector}')
            plt.grid(True)
            plt.legend()
            plt.show()


        # write results to CSV
        results = {
            'Stock_Name' : stock_name,
            'Feature_Vector' : feature_vector,
            'Train_Ratio' : train_ratio,
            'MSE' : MSE,
            'MAPE' : MAPE#,
            # 'Test_Data' : test_data.to_list(),
            # 'Predicted_Data' : pred.to_list()
        }
        all_results.append(results)
        
        tqdm.tqdm.write(f"Processed {stock_name}")

    file_name = f'arima_{feature_vector}.csv'
    csv_header = ['Stock_Name', 'Feature_Vector', 'Train_Ratio', 'MSE', 'MAPE'] # , 'Test_Data', 'Predicted_Data']

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)


# Function to visualize test MSE at different training ratios
# This was used to check what ratio is best to prevent overfitting
def ARIMA_at_various_train_ratios(data, train_ratio_range, feature_vector, plot_mode=False):
    for i in tqdm.tqdm(train_ratio_range, desc="ARIMA_at_various_train_ratios"):
        # print(i)
        ARIMA_MSE_and_write_to_csv(data, feature_vector, i)

    results_df = pd.DataFrame(all_results)

    if plot_mode:
        # Plotting MSE and MAPE vs training ratios
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.plot(results_df['Train_Ratio'], results_df['MSE'], marker='o', linestyle='-', color='b')
        plt.title('MSE vs Training Ratio')
        plt.xlabel('Training Ratio')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(results_df['Train_Ratio'], results_df['MAPE'], marker='o', linestyle='-', color='r')
        plt.title('MAPE vs Training Ratio')
        plt.xlabel('Training Ratio')
        plt.ylabel('Mean Absolute Percentage Error (MAPE)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# Execution function
def main():
    # train_ratio_range = np.arange(0.1, 1.0, 0.01)
    # ARIMA_at_various_train_ratios(stock_data, train_ratio_range, 'Close', plot_mode=True)
    # ARIMA_MSE_and_write_to_csv(stock_data, 'Close', 0.9, plot_mode=True)
    ARIMA_MSE_and_write_to_csv(stock_data, 'Close', 0.8, plot_mode=True)
    # ARIMA_MSE_and_write_to_csv(stock_data, 'Close', 0.5, plot_mode=True)
    # ARIMA_MSE_and_write_to_csv(stock_data, 'Close', 0.2, plot_mode=True)
main()