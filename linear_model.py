import pandas as pd  # Import pandas to read the dataset
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset from the same folder
dataset_path = "sample_stock_crypto_data.csv"  # File is in the same directory
df = pd.read_csv(dataset_path)

# Convert Date column to datetime format and keep it in the DataFrame
df["Date"] = pd.to_datetime(df["Date"])

# Fill missing values with the previous day's price
df["Close"].fillna(method="ffill", inplace=True)

# Create technical indicators
df["Previous_Close"] = df["Close"].shift(1)  # Previous day's close
df["SMA_5"] = df["Close"].rolling(window=5).mean()  # 5-day Simple Moving Average
df["SMA_10"] = df["Close"].rolling(window=10).mean()  # 10-day Simple Moving Average
df["Price_Change"] = df["Close"].pct_change()  # Daily percentage change
df["Mean_Reversion"] = df["Close"] - df["Close"].rolling(window=10).mean()  # Regression to the mean indicator
df["High_Low_Diff"] = df["High"] - df["Low"]  # Difference between High and Low prices
df["Volatility"] = df["Close"].rolling(5).std()  # 5-day Rolling Standard Deviation for volatility

# Calculate RSI (Relative Strength Index)
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))
df["RSI"] = df["RSI"].replace([np.inf, -np.inf], np.nan)  # Remove infinite values
df["RSI"].fillna(50, inplace=True)  # Ensure RSI is never empty

df.dropna(inplace=True)  # Remove any remaining NaN values

# Print RSI values in the terminal
print("\nRSI Values:")
print(df[["Date", "RSI"]].tail(20))

# Prepare data for training
X = df[["Previous_Close", "SMA_5", "SMA_10", "Price_Change", "Mean_Reversion", "High_Low_Diff", "Volatility", "RSI"]]  # Features
y = df["Close"]  # Target variable

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Train a Linear Regression model for comparison
lr_model = LinearRegression()
lr_model.fit(X, y)

# Make predictions
df["RF_Predicted_Close"] = rf_model.predict(X)
df["LR_Predicted_Close"] = lr_model.predict(X)

# Calculate model performance metrics
rf_mae = mean_absolute_error(y, df["RF_Predicted_Close"])
rf_mse = mean_squared_error(y, df["RF_Predicted_Close"])
rf_r2 = r2_score(y, df["RF_Predicted_Close"])

lr_mae = mean_absolute_error(y, df["LR_Predicted_Close"])
lr_mse = mean_squared_error(y, df["LR_Predicted_Close"])
lr_r2 = r2_score(y, df["LR_Predicted_Close"])

print("\nRandom Forest Model Performance:")
print(f"MAE: {rf_mae:.2f}, MSE: {rf_mse:.2f}, R²: {rf_r2:.4f}")

print("\nLinear Regression Model Performance:")
print(f"MAE: {lr_mae:.2f}, MSE: {lr_mse:.2f}, R²: {lr_r2:.4f}")

# Print actual vs predicted prices in the terminal
print("\nActual vs Predicted Prices:")
print(df[["Date", "Close", "RSI", "RF_Predicted_Close", "LR_Predicted_Close"]].tail(20))

# Create Simple UI for Viewing Predictions
def show_predictions():
    window = tk.Tk()
    window.title("Stock Price Predictions")
    window.geometry("900x500")
    
    tree = ttk.Treeview(window, columns=("Date", "Actual", "RSI", "RF Prediction", "LR Prediction"), show="headings")
    tree.heading("Date", text="Date")
    tree.heading("Actual", text="Actual Price")
    tree.heading("RSI", text="RSI")
    tree.heading("RF Prediction", text="RF Prediction")
    tree.heading("LR Prediction", text="LR Prediction")
    
    for _, row in df.tail(20).iterrows():
        rsi_value = "N/A" if pd.isna(row["RSI"]) else round(row["RSI"], 2)
        tree.insert("", "end", values=(
            row["Date"].strftime("%Y-%m-%d"), 
            round(row["Close"], 2), 
            rsi_value, 
            round(row["RF_Predicted_Close"], 2), 
            round(row["LR_Predicted_Close"], 2)
        ))
    
    tree.pack(expand=True, fill="both")
    window.mainloop()

show_predictions()
