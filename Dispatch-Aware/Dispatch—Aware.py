# =========================
# Dispatch-Aware vs Standard Forecast Comparison
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# =========================================================
# 1. Load data
# =========================================================
file_path = 'load_data.csv'

data = pd.read_csv(
    file_path,
    parse_dates=['Datetime'],
    index_col='Datetime'
).sort_index()

# Handle possible duplicate timestamps
if data.index.duplicated().any():
    print("Duplicate timestamps detected. Aggregating by mean.")
    data = data.groupby(data.index).mean()

load = data['AEP_MW']

# =========================================================
# 2. Feature engineering
# =========================================================
df = pd.DataFrame({'load': load})
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['dayofyear'] = df.index.dayofyear
df['lag_24'] = df['load'].shift(24)
df['lag_168'] = df['load'].shift(168)
df = df.dropna()

features = ['hour', 'dayofweek', 'month', 'dayofyear', 'lag_24', 'lag_168']
X = df[features]
y = df['load']

# =========================================================
# 3. Train / test split
# =========================================================
TEST_HOURS = 5000
X_train = X[:-TEST_HOURS]
y_train = y[:-TEST_HOURS]
X_test = X[-TEST_HOURS:]
y_test = y[-TEST_HOURS:]

# =========================================================
# 4. Train standard XGBoost model
# =========================================================
print("Training standard XGBoost model...")
standard_model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    objective='reg:squarederror',
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
standard_model.fit(X_train, y_train)
standard_pred = standard_model.predict(X_test)

# =========================================================
# 5. Apply dispatch-aware bias (1.03)
# =========================================================
DISPATCH_BIAS = 1.03
dispatch_pred = standard_pred * DISPATCH_BIAS

# =========================================================
# 6. Evaluation metrics
# =========================================================
rmse_standard = np.sqrt(mean_squared_error(y_test, standard_pred))
rmse_dispatch = np.sqrt(mean_squared_error(y_test, dispatch_pred))
mean_error_standard = np.mean(standard_pred - y_test)
mean_error_dispatch = np.mean(dispatch_pred - y_test)

RESERVE_CAPACITY = 2000
shortage_standard = np.sum(y_test > standard_pred + RESERVE_CAPACITY)
shortage_dispatch = np.sum(y_test > dispatch_pred + RESERVE_CAPACITY)

print("\nEvaluation Results (Bias = 1.03):")
print(f"Standard RMSE           : {rmse_standard:.2f}")
print(f"Dispatch-aware RMSE     : {rmse_dispatch:.2f}")
print(f"Standard Mean Error     : {mean_error_standard:.2f}")
print(f"Dispatch-aware Mean Error: {mean_error_dispatch:.2f}")
print(f"Reserve shortage (standard)     : {shortage_standard}")
print(f"Reserve shortage (dispatch-aware): {shortage_dispatch}")

# =========================================================
# 7. Visualization (last 200 hours)
# =========================================================
PLOT_HOURS = 200
actual = y_test[-PLOT_HOURS:].values
std_fcst = standard_pred[-PLOT_HOURS:]
disp_fcst = dispatch_pred[-PLOT_HOURS:]

plt.figure(figsize=(14, 7))
plt.plot(actual, label='Actual Load', linewidth=2, color='blue')
plt.plot(std_fcst, label='Standard Forecast', alpha=0.8, color='green')
plt.plot(disp_fcst, label='Dispatch-Aware Forecast (+3%)', linewidth=3, color='red')
plt.title('Standard vs Dispatch-Aware Load Forecasting (Last 200 Hours)')
plt.xlabel('Time')
plt.ylabel('Load (MW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
