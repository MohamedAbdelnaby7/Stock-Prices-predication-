"""
Created on Thu Dec 07:02:03 2023

@author: Mohamed Abdelnaby - Ph.D. Candidate at BU mabnaby@bu.edu
 Version 1.5
'''
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings

# To filter out all warnings
warnings.filterwarnings("ignore")

# Load stock price data from CSV files in the ".\stocks" directory
stocks_folder = ".\stocks"
stock_files = [file for file in os.listdir(stocks_folder) if file.endswith(".csv")]

# Initialize RLS parameters
lambda_factor = 0.98  # Forgetting factor
n_features = 4

# Function for RLS algorithm
def rls_algorithm(data):
    P = lambda_factor * np.identity(n_features)
    theta = np.zeros(n_features)
    predicted_prices_rls = []

    for _, row in data.iterrows():
        x = np.array([row['Open'], row['High'], row['Low'], row['Volume']])
        y = row['Close']

        # RLS algorithm
        error = y - np.dot(theta, x)
        P = (1/lambda_factor) * (P - np.outer(P.dot(x), P.dot(x).T) / (lambda_factor + x.T.dot(P).dot(x)))
        theta = theta + P.dot(x) * error

        # Predict the next stock price
        predicted_price = np.dot(theta, x)
        predicted_prices_rls.append(predicted_price)

    return predicted_prices_rls, theta

# Function to calculate MSE and plot results
def evaluate_model(ax, data, title, predicted_prices_rls_train, predicted_prices_rls_test, linear_predictions_train, linear_predictions_test, time_indices, train_size):
    if isinstance(ax, np.ndarray):  # Check if ax is an array
        # If ax is an array, select the first subplot
        ax = ax[0, 0]
    # Calculate Mean Squared Errors on the training set
    mse_linear_train = mean_squared_error(data['Close'][:train_size], linear_predictions_train)
    mse_rls_train = mean_squared_error(data['Close'][:train_size], predicted_prices_rls_train)

    # Calculate Mean Squared Errors on the testing set
    mse_linear_test = mean_squared_error(data['Close'][train_size:], linear_predictions_test)
    mse_rls_test = mean_squared_error(data['Close'][train_size:], predicted_prices_rls_test)

    # Plot actual and predicted prices
    ax.plot(time_indices, data['Close'], label='Actual Close Price')
    ax.plot(time_indices[:train_size], predicted_prices_rls_train, label='RLS - Training')
    ax.plot(time_indices[:train_size], linear_predictions_train, label='Linear Reg - Training')
    ax.plot(time_indices[train_size:], predicted_prices_rls_test, label='RLS - Testing')
    ax.plot(time_indices[train_size:], linear_predictions_test, label='Linear Reg - Testing')
    ax.set_title(title)
    ax.legend(fontsize='small')

    # Annotate the plot with MSE values for training and testing sets
    ax.annotate(f'MSE Linear (Train): {mse_linear_train:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', color='green')
    ax.annotate(f'MSE RLS (Train): {mse_rls_train:.2f}', xy=(0.05, 0.80), xycoords='axes fraction', color='orange')
    ax.annotate(f'MSE Linear (Test): {mse_linear_test:.2f}', xy=(0.05, 0.75), xycoords='axes fraction', color='purple')
    ax.annotate(f'MSE RLS (Test): {mse_rls_test:.2f}', xy=(0.05, 0.70), xycoords='axes fraction', color='red')
    # Return the MSE values
    return {
        'Stock': title,
        'MSE Linear (Train)': mse_linear_train,
        'MSE RLS (Train)': mse_rls_train,
        'MSE Linear (Test)': mse_linear_test,
        'MSE RLS (Test)': mse_rls_test
    }

stop_learning = False
for j in range(2):
    # Create subplots for each stock
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Stock Price Predictions', fontsize=16)
    # Create a list to store MSE values for all stocks
    mse_values_all = []
    stop_learning = ~stop_learning
    for i, stock_file in enumerate(stock_files[:9]):
        file_path = os.path.join(stocks_folder, stock_file)
        data_stock = pd.read_csv(file_path)
    
        # Split the data into training and testing sets
        train_data, test_data = train_test_split(data_stock, test_size=0.2, shuffle=False)
    
        # Apply RLS algorithm on the training set
        predicted_prices_rls_train, theta_train = rls_algorithm(train_data)
        if ~stop_learning:
            # Apply RLS algorithm on the testing set using the learned theta
            predicted_prices_rls_test, _ = rls_algorithm(test_data)
        else:
            x = np.array([test_data['Open'], test_data['High'], test_data['Low'], test_data['Volume']])
            predicted_prices_rls_test = np.dot(theta_train, x)
    
        # Fit a linear regression model as a threshold
        X_train = train_data[['Open', 'High', 'Low', 'Volume']]
        y_train = train_data['Close']
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        linear_predictions_train = linear_model.predict(X_train)
    
        # Predictions on the testing set using the linear model
        X_test = test_data[['Open', 'High', 'Low', 'Volume']]
        linear_predictions_test = linear_model.predict(X_test)
    
        r, c = divmod(i, 3)
        print(r, c)
        # Evaluate the models and plot results in the respective subplot
        mse_values = evaluate_model(axs[r,c], data_stock, f'Stock: {stock_file.replace(".csv", "")}', predicted_prices_rls_train, predicted_prices_rls_test, linear_predictions_train, linear_predictions_test, data_stock.index, len(train_data))
        # Append MSE values to the list
        mse_values_all.append(mse_values)
    # Convert the list of MSE values to a DataFrame
    mse_df_all = pd.DataFrame(mse_values_all)# Display the MSE DataFrame
    mse_df_all.to_csv(f'mse_df_all_iteration_{j}.csv', index=False)
    fig.show()

# Initialize RLS variables for transfer learning
P_transfer = lambda_factor * np.identity(n_features)
theta_transfer = theta_train.copy()  # Initialize with the learned theta from training

# Choose another stock for transfer learning (let's say the 16th stock)
transfer_stock_file = stock_files[15]
file_path_transfer = os.path.join(stocks_folder, transfer_stock_file)
data_transfer = pd.read_csv(file_path_transfer)

# Initialize lists to store true and predicted prices
true_prices_transfer = []
predicted_prices_transfer = []

# 'data_transfer' is Pandas DataFrame with columns 'Open', 'High', 'Low', 'Close', 'Volume'
for index, row in data_transfer.iterrows():
    x_transfer = np.array([row['Open'], row['High'], row['Low'], row['Volume']])
    y_transfer = row['Close']

    # RLS algorithm for transfer learning
    error_transfer = y_transfer - np.dot(theta_transfer, x_transfer)
    P_transfer = (1/lambda_factor) * (P_transfer - np.outer(P_transfer.dot(x_transfer), P_transfer.dot(x_transfer).T) / (lambda_factor + x_transfer.T.dot(P_transfer).dot(x_transfer)))
    theta_transfer = theta_transfer + P_transfer.dot(x_transfer) * error_transfer

    # Predict the next stock price during transfer learning
    predicted_price_transfer = np.dot(theta_transfer, x_transfer)

    # Store true and predicted prices
    true_prices_transfer.append(y_transfer)
    predicted_prices_transfer.append(predicted_price_transfer)

# Calculate MSE for transfer learning
mse_transfer = mean_squared_error(true_prices_transfer, predicted_prices_transfer)
print(f'Mean Squared Error (Transfer Learning): {mse_transfer}')

# Plot the results
fig_transfer, ax_transfer = plt.subplots(figsize=(10, 6))
ax_transfer.plot(data_transfer['Close'], label='Actual Close Price (Transfer)')
ax_transfer.plot(predicted_prices_transfer, label='Predicted Close Price (Transfer)')
ax_transfer.set_title(f'Stock: {transfer_stock_file.replace(".csv", "")} - Transfer Learning')
ax_transfer.legend()
plt.show()