# =========================
# Sensitivity Analysis: Trade-off Curve
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# =========================================================
# 1. Load data and train baseline model (same as before)
# =========================================================
file_path = 'load_data.csv'

data = pd.read_csv(
    file_path,
    parse_dates=['Datetime'],
    index_col='Datetime'
).sort_index()

if data.index.duplicated().any():
    print("Duplicate timestamps detected. Aggregating by mean.")
    data = data.groupby(data.index).mean()

load = data['AEP_MW']

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

TEST_HOURS = 5000
X_train = X[:-TEST_HOURS]
y_train = y[:-TEST_HOURS]
X_test = X[-TEST_HOURS:]
y_test = y[-TEST_HOURS:]

print("Training standard XGBoost model for sensitivity analysis...")
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
# 2. Sensitivity analysis
# =========================================================
biases = [1.00, 1.01, 1.02, 1.03, 1.05]
RESERVE_CAPACITY = 2000

results = []

print("\n" + "=" * 60)
print("Sensitivity Analysis: Impact of Different Conservative Biases")
print("=" * 60)

for bias in biases:
    dispatch_pred = standard_pred * bias
    rmse = np.sqrt(mean_squared_error(y_test, dispatch_pred))
    mean_error = np.mean(dispatch_pred - y_test)
    shortage = np.sum(y_test > dispatch_pred + RESERVE_CAPACITY)

    results.append({
        'Bias': f"{bias:.2f}",
        'RMSE': round(rmse, 2),
        'Mean Error': round(mean_error, 2),
        'Shortage Events': shortage
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# =========================================================
# 3. Trade-off curve
# =========================================================
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Conservative Bias', fontsize=12)
ax1.set_ylabel('RMSE (MW)', color='tab:blue', fontsize=12)
ax1.plot(results_df['Bias'], results_df['RMSE'],
         marker='o', markersize=8, color='tab:blue', linewidth=2.5, label='RMSE')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.set_ylabel('Reserve Shortage Events', color='tab:red', fontsize=12)
ax2.plot(results_df['Bias'], results_df['Shortage Events'],
         marker='s', markersize=8, color='tab:red', linewidth=2, label='Shortage Events')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Trade-off Curve: Forecast Conservativeness vs Operational Risk', fontsize=14, pad=20)
fig.tight_layout()
plt.show()
