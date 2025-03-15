import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# File path
file_path = '/Users/lihanyu/Desktop/NLP-Final/Dataset/Stock.csv'
stock_data = pd.read_csv(file_path)

# Convert 'Date' to datetime and sort
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.sort_values(by='Date', inplace=True)

# Map sentiment values: Neutral → 0, Negative → -1, Positive → 1
stock_data['stock sentiment'] = stock_data['stock sentiment'].map({
    'Neutral': 0, 
    'Negative': -1, 
    'Positive': 1
})

# Features and target column
features = ['Volume', 'daily_volatility', 'nlp_daily_sentiment', 'price_range', 'gap', 'stock sentiment']
target = 'Close/Last'

# Convert 'Close/Last' to float
stock_data[target] = stock_data[target].replace('[\$,]', '', regex=True).astype(float)

# Drop rows with missing values in the relevant columns
stock_data = stock_data.dropna(subset=features + [target])

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(stock_data[features + [target]])

# Create time series sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-1])  # Features
        y.append(data[i + sequence_length, -1])    # Target (Close/Last)
    return np.array(X), np.array(y)

sequence_length = 30  # Use past 30 days to predict the next day
X, y = create_sequences(scaled_data, sequence_length)

# Split into training and testing sets
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # Only take the last hidden state
        out = self.fc(hidden[-1])     # Pass through the fully connected layer
        return out

# Model parameters
input_size = X_train.shape[2]
hidden_size = 50
num_layers = 2
output_size = 1

# Initialize model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0.0
    
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(X_train):.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test.numpy(), y_pred.numpy())
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

# Rescale predictions and true values
y_test_rescaled = scaler.inverse_transform(
    np.concatenate((np.zeros((len(y_test), X_test.shape[2])), y_test.numpy().reshape(-1, 1)), axis=1)
)[:, -1]

y_pred_rescaled = scaler.inverse_transform(
    np.concatenate((np.zeros((len(y_pred), X_test.shape[2])), y_pred.numpy().reshape(-1, 1)), axis=1)
)[:, -1]

# Plot true vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='True Close Price', marker='o', alpha=0.7)
plt.plot(y_pred_rescaled, label='Predicted Close Price', marker='x', alpha=0.7)
plt.title('True vs Predicted Close Prices')
plt.xlabel('Time Steps')
plt.ylabel('Close Price')
plt.legend()
plt.grid(alpha=0.5)
plt.show()
