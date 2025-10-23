import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# ============== LOAD & CLEAN DATA ==================
DATA_FILE = "AAVEUSD_day.csv"
df = pd.read_csv(DATA_FILE, skipinitialspace=True)

df.columns = df.columns.str.strip()
if 'date' not in df.columns:
    for col in df.columns:
        if 'date' in col.lower():
            df.rename(columns={col: 'date'}, inplace=True)
            break

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print("✅ Data loaded successfully!")
print(df.head())

# ============== CREATE ROLLING WINDOWS =============
LOOKBACK = 60
features, labels = [], []

for i in range(LOOKBACK, len(df)):
    window = df['close'].values[i-LOOKBACK:i]
    features.append(window)
    labels.append(df['close'].values[i])

features = np.array(features)
labels = np.array(labels)

print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

# ============== SCALE DATA ========================
scaler_X = MinMaxScaler()
features_scaled = scaler_X.fit_transform(features)
features_scaled = features_scaled[..., np.newaxis]  # add (samples, timesteps, features)

scaler_y = MinMaxScaler()
labels_scaled = scaler_y.fit_transform(labels.reshape(-1,1)).flatten()

# ============== BUILD & TRAIN LSTM ================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK,1)),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

print("🚀 Training the model...")
history = model.fit(
    features_scaled, labels_scaled,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# ============== MAKE PREDICTIONS ==================
pred_scaled = model.predict(features_scaled)
predicted = scaler_y.inverse_transform(pred_scaled)
actual = df['close'].values[LOOKBACK:]

# ============== VISUALIZE =========================
plt.figure(figsize=(12,6))
plt.plot(df['date'].values[LOOKBACK:], actual, label="Actual")
plt.plot(df['date'].values[LOOKBACK:], predicted, label="Predicted")
plt.title("AAVE/USD — Actual vs Predicted Prices")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()

# Save the plot instead of showing it
plt.tight_layout()
plt.savefig("prediction_plot.png", dpi=300)
plt.close()

print("📊 Plot saved successfully as 'prediction_plot.png'")
# Save trained model and scalers for later use
model.save("trained_lstm_model.h5")
import joblib
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

print("✅ Model and scalers saved successfully.")

