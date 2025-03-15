from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the stock data from the specified file path
stock_file_path = '/Users/lihanyu/Desktop/NLP-Final/Dataset/Stock.csv'
stock_data = pd.read_csv(stock_file_path)

# Preprocess the data
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.sort_values(by='Date', inplace=True)

# Select features and target variable
features = ['Volume', 'daily_volatility', 'nlp_daily_sentiment', 'price_range', 'gap']
target = 'Close/Last'

stock_data[target] = stock_data[target].replace('[\$,]', '', regex=True).astype(float)

# Drop rows with missing values in features or target
stock_data = stock_data.dropna(subset=features + [target])

# Split the data into train and test sets
X = stock_data[features]
y = stock_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Visualization 1: True vs Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.title('True vs Predicted Values', fontsize=14)
plt.xlabel('True Close/Last', fontsize=12)
plt.ylabel('Predicted Close/Last', fontsize=12)
plt.grid(alpha=0.5)
plt.show()

# Visualization 2: Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.title('Residual Plot', fontsize=14)
plt.xlabel('Predicted Close/Last', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.grid(alpha=0.5)
plt.show()

# Visualization 3: Feature Importance
importances = rf_model.feature_importances_
feature_names = features

plt.figure(figsize=(8, 6))
plt.barh(feature_names, importances, alpha=0.7)
plt.title('Feature Importance', fontsize=14)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.grid(axis='x', alpha=0.5)
plt.show()

# Visualization 4: Time Series of Predictions vs True Values
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Date'][-len(y_test):], y_test, label='True Values', marker='o', alpha=0.7)
plt.plot(stock_data['Date'][-len(y_test):], y_pred, label='Predicted Values', marker='x', alpha=0.7)
plt.title('True vs Predicted Close Prices Over Time', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close/Last', fontsize=12)
plt.legend()
plt.grid(alpha=0.5)
plt.show()
