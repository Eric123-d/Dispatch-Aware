# =========================================================
# Dispatch-Aware Load Forecasting with Failure Monitoring
# =========================================================
#
# This script implements a two-layer forecasting architecture:
#
# Layer 1: Forecasting Layer
#   - Machine learning-based short-term load forecasting (XGBoost)
#   - Time-aware feature engineering with historical memory
#   - Optional dispatch-aware bias to reflect conservative operation
#
# Layer 2: Failure Monitoring System (FMS)
#   - Non-intrusive safety and observability layer
#   - Does NOT modify or correct forecasts
#   - Does NOT interfere with model training or inference
#   - Explicitly detects and exposes unsafe or abnormal conditions
#
# Core Engineering Philosophy:
#   Forecasts are allowed to be wrong.
#   Failures must never be silent.
#
# The objective is not zero prediction error,
# but transparent risk awareness under extreme or unsafe conditions.
#
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# =========================================================
# 1. Data ingestion and integrity handling
# =========================================================
# - Load historical system load data
# - Enforce chronological ordering
# - Resolve duplicated timestamps explicitly via aggregation
# - Preserve data uncertainty rather than silently discarding it

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

# =========================================================
# 2. Feature engineering (time-aware + memory effects)
# =========================================================
# Feature design reflects:
# - Intra-day cyclic behavior (hour)
# - Weekly operational patterns (day of week)
# - Seasonal variation (month, day of year)
# - Short-term and weekly persistence (24h / 168h lag)
#
# No exogenous variables are assumed.
# The model is intentionally constrained to historical load signals
# to keep failure modes interpretable.

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
# 3. Temporal train / test separation
# =========================================================
# - Strict time-based split
# - Prevents information leakage
# - Test horizon represents unseen operational future

TEST_HOURS = 5000
X_train = X[:-TEST_HOURS]
y_train = y[:-TEST_HOURS]
X_test = X[-TEST_HOURS:]
y_test = y[-TEST_HOURS:]

# =========================================================
# 4. Baseline forecasting model (XGBoost)
# =========================================================
# - Gradient boosted decision trees
# - Optimized for nonlinear temporal interactions
# - Trained purely as a predictor, without safety constraints
#
# The model is treated as a black-box forecaster.

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
# 5. Dispatch-aware bias layer
# =========================================================
# A fixed upward bias is applied to reflect conservative
# operational decision-making.
#
# This is NOT model retraining.
# This is NOT post-hoc error correction.
# This represents a dispatch-side policy choice.
#
# Forecasting and dispatch decisions are intentionally decoupled.

DISPATCH_BIAS = 1.03
dispatch_pred = standard_pred * DISPATCH_BIAS

# =========================================================
# 6. Forecast accuracy and operational risk metrics
# =========================================================
# Metrics include:
# - RMSE: statistical accuracy
# - Mean error: systematic bias
# - Reserve shortage count: operational risk indicator
#
# Reserve shortage does NOT imply model failure.
# It indicates potential system stress under forecast uncertainty.

rmse_standard = np.sqrt(mean_squared_error(y_test, standard_pred))
rmse_dispatch = np.sqrt(mean_squared_error(y_test, dispatch_pred))
mean_error_standard = np.mean(standard_pred - y_test)
mean_error_dispatch = np.mean(dispatch_pred - y_test)

RESERVE_CAPACITY = 2000
shortage_standard = np.sum(y_test > standard_pred + RESERVE_CAPACITY)
shortage_dispatch = np.sum(y_test > dispatch_pred + RESERVE_CAPACITY)

print("\nEvaluation Results (Bias = 1.03):")
print(f"Standard RMSE            : {rmse_standard:.2f}")
print(f"Dispatch-aware RMSE      : {rmse_dispatch:.2f}")
print(f"Standard Mean Error      : {mean_error_standard:.2f}")
print(f"Dispatch-aware Mean Error: {mean_error_dispatch:.2f}")
print(f"Reserve shortage (standard)      : {shortage_standard}")
print(f"Reserve shortage (dispatch-aware): {shortage_dispatch}")

# =========================================================
# 7. Visualization (human-in-the-loop inspection)
# =========================================================
# Last 200 hours are visualized to:
# - Compare baseline vs dispatch-aware forecasts
# - Enable qualitative inspection of forecast behavior
# - Support post-event and operational analysis

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

# =========================================================
# 8. Failure Monitoring System (FMS)
# =========================================================
#
# The FMS is a non-intrusive safety and observability layer.
#
# Core assumptions:
# - Forecasting errors are inevitable
# - Extreme or adversarial conditions cannot be eliminated
# - Silent failures are unacceptable
#
# Therefore, the FMS:
# - Does NOT modify forecasts
# - Does NOT attempt self-correction
# - Explicitly surfaces unsafe or abnormal conditions
#
# The system prioritizes transparency over optimism.

# -------------------------
# 8.1 Explicit failure definitions
# -------------------------
# Failure thresholds are intentionally conservative and interpretable.
# They are NOT optimized for alert frequency,
# but chosen to reflect operationally meaningful risk.

ABS_ERROR_THRESHOLD = 3000        # Absolute forecast error (MW)
REL_ERROR_THRESHOLD = 0.08        # Relative error (% of actual load)
DISPATCH_RISK_MARGIN = RESERVE_CAPACITY

# -------------------------
# 8.2 Error signal extraction
# -------------------------
# Raw error signals are preserved.
# No smoothing, filtering, or suppression is applied.
#
# The goal is observability, not noise reduction.

abs_error = np.abs(dispatch_pred - y_test)
rel_error = abs_error / y_test

# -------------------------
# 8.3 Failure mode evaluation
# -------------------------
# Each failure mode is evaluated independently.
# A single triggered mode is sufficient to raise an alert.
#
# This avoids masking compound or correlated risks.

failure_abs = abs_error > ABS_ERROR_THRESHOLD
failure_rel = rel_error > REL_ERROR_THRESHOLD
failure_dispatch = y_test > (dispatch_pred + DISPATCH_RISK_MARGIN)

fms_alert = failure_abs | failure_rel | failure_dispatch

# -------------------------
# 8.4 Failure transparency and statistics
# -------------------------
# Summary statistics provide:
# - Overall system stability
# - Failure mode distribution
# - Alert frequency
#
# Low alert rates are expected and desirable.

total_points = len(y_test)
total_alerts = np.sum(fms_alert)

print("\nFailure Monitoring System (FMS) Summary:")
print(f"Total evaluated points     : {total_points}")
print(f"Total FMS alerts triggered : {total_alerts}")
print(f"Alert rate (%)             : {100 * total_alerts / total_points:.2f}")

print("\nFMS Failure Breakdown:")
print(f"Absolute error failures : {np.sum(failure_abs)}")
print(f"Relative error failures : {np.sum(failure_rel)}")
print(f"Dispatch risk failures  : {np.sum(failure_dispatch)}")

# -------------------------
# 8.5 Transparent failure logging
# -------------------------
# A structured failure log is generated to:
# - Support forensic analysis
# - Enable threshold sensitivity studies
# - Preserve accountability of decisions
#
# No failures are discarded or overwritten.

fms_log = pd.DataFrame({
    'Actual_Load': y_test.values,
    'Dispatch_Forecast': dispatch_pred,
    'Absolute_Error': abs_error,
    'Relative_Error': rel_error,
    'Abs_Error_Failure': failure_abs,
    'Rel_Error_Failure': failure_rel,
    'Dispatch_Risk_Failure': failure_dispatch,
    'FMS_Alert': fms_alert
}, index=y_test.index)

top_failures = fms_log.sort_values(
    by='Absolute_Error', ascending=False
).head(10)

print("\nTop 10 worst forecast failures (by absolute error):")
print(top_failures[['Actual_Load', 'Dispatch_Forecast', 'Absolute_Error']])

# -------------------------
# 8.6 Recent horizon alert inspection
# -------------------------
# Recent alerts are inspected separately to evaluate
# near-term operational exposure.
#
# Visual markers ensure alerts are human-visible,
# not just numerically counted.

recent_alerts = fms_log[-PLOT_HOURS:]
num_recent_alerts = recent_alerts['FMS_Alert'].sum()

print(f"\nFMS alerts in last {PLOT_HOURS} hours: {num_recent_alerts}")

alert_indices = recent_alerts[recent_alerts['FMS_Alert']].index
alert_actual = recent_alerts.loc[alert_indices, 'Actual_Load']

plt.scatter(
    range(len(alert_indices)),
    alert_actual,
    color='black',
    label='FMS Alert',
    zorder=5
)
plt.legend()
plt.show()
