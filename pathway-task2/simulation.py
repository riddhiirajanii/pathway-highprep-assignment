import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# -----------------------
# Load model & scalers
# -----------------------
model = load_model("trained_lstm_model.h5", compile=False)
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# -----------------------
# Load dataset
# -----------------------
DATA_FILE = "AAVEUSD_day.csv"
LOOKBACK = 60
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# -----------------------
# Simulate real-time market
# -----------------------
predicted_prices = []
actual_prices = []
dates = []

for i in range(LOOKBACK, len(df)):
    window = df['close'].iloc[i-LOOKBACK:i].values.reshape(1, -1)
    window_scaled = scaler_X.transform(window)
    window_scaled = window_scaled[..., np.newaxis]

    pred_scaled = model.predict(window_scaled, verbose=0)[0][0]
    pred_unscaled = scaler_y.inverse_transform([[pred_scaled]])[0][0]

    predicted_prices.append(pred_unscaled)
    actual_prices.append(df['close'].iloc[i])
    dates.append(df['date'].iloc[i])

# -----------------------
# Evaluate simulation
# -----------------------
actual_prices = np.array(actual_prices)
predicted_prices = np.array(predicted_prices)

# Directional Accuracy
correct = np.sum(np.sign(np.diff(actual_prices)) == np.sign(np.diff(predicted_prices)))
directional_accuracy = correct / (len(actual_prices) - 1)
print(f"Directional Accuracy: {directional_accuracy*100:.2f}%")

# -----------------------
# Generate trading signals
# -----------------------
signals = np.sign(np.diff(predicted_prices, prepend=predicted_prices[0]))
# -----------------------
# Backtesting: Simple Profit Simulation
# -----------------------
initial_balance = 10000  # starting USD
balance = initial_balance
position = 0  # 0 = no position, 1 = holding one unit of asset
portfolio_values = []

for i in range(1, len(predicted_prices)):
    price = actual_prices[i]
    prev_signal = signals[i-1]

    # Buy if signal = 1 and we are not already holding
    if prev_signal == 1 and position == 0:
        position = 1
        buy_price = price

    # Sell if signal = -1 and we are holding
    elif prev_signal == -1 and position == 1:
        balance += (price - buy_price)
        position = 0

    # Track portfolio value (cash + asset value)
    portfolio_value = balance + (price - buy_price if position == 1 else 0)
    portfolio_values.append(portfolio_value)

final_value = portfolio_values[-1]
profit = final_value - initial_balance
returns = (profit / initial_balance) * 100

print(f"Final Portfolio Value: ${final_value:.2f}")
print(f"Total Profit: ${profit:.2f} ({returns:.2f}%)")
plt.figure(figsize=(10,5))
plt.plot(dates[1:], portfolio_values, color='green', label='Portfolio Value')
plt.title("Simulated Trading Performance")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("out/trading_performance.png")
print("Trading performance plot saved as out/trading_performance.png")
# -----------------------
# Plot results
# -----------------------
os.makedirs("out", exist_ok=True)
plt.figure(figsize=(12,6))
plt.plot(dates, actual_prices, label="Actual Close Price", color="blue")
plt.plot(dates, predicted_prices, label="Predicted Close Price", color="orange")
plt.title("AAVE/USD Market Simulation - Predicted vs Actual Prices")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("out/market_simulation.png")
print("Market simulation complete. Plot saved as out/market_simulation.png")
