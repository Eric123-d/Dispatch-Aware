# =========================
# XGBoost ONLY – Continuous Visualization (零警告版)
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# =========================
# 1. Load data
# =========================
file_path = 'load_data.csv'

data = pd.read_csv(
    file_path,
    parse_dates=['Datetime'],
    index_col='Datetime'
).sort_index()

load = data['AEP_MW']

# =========================
# 关键修复：处理重复索引（如果有）
# =========================
if load.index.duplicated().any():
    print("检测到重复时间索引，正在取平均值聚合...")
    load = load.groupby(load.index).mean()

# =========================
# 2. Feature engineering
# =========================
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

# =========================
# 3. Train XGBoost
# =========================
print("Training XGBoost model...")
X_train = X[:-24]
y_train = y[:-24]

xgb_model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    objective='reg:squarederror',
    random_state=42
)

xgb_model.fit(X_train, y_train)

# =========================
# 4. Recursive 24h forecast（正确更新所有时间特征）
# =========================
print("Generating XGBoost recursive 24-hour forecast...")
xgb_forecast = []

# 从最后一个已知时间点开始
current_time = df.index[-1]
last_lag_24 = df['lag_24'].iloc[-1]
last_lag_168 = df['lag_168'].iloc[-1]

for i in range(24):
    # 计算未来第 (i+1) 小时的时间点
    future_time = current_time + pd.Timedelta(hours=i + 1)

    # 创建特征
    feature_row = pd.DataFrame([{
        'hour': future_time.hour,
        'dayofweek': future_time.dayofweek,
        'month': future_time.month,
        'dayofyear': future_time.dayofyear,
        'lag_24': last_lag_24,
        'lag_168': last_lag_168
    }])

    # 预测
    pred = xgb_model.predict(feature_row)[0]
    xgb_forecast.append(pred)

    # 更新滞后特征（递归）
    last_lag_168 = last_lag_24
    last_lag_24 = pred

# 生成未来索引（使用小写 'h'，消除警告）
xgb_index = pd.date_range(
    start=load.index[-1] + pd.Timedelta(hours=1),
    periods=24,
    freq='h'  # ← 关键：改为小写 'h'
)

xgb_forecast = pd.Series(xgb_forecast, index=xgb_index)

# =========================
# 5. 关键：让预测线从历史最后一个点完美接上
# =========================
xgb_plot_series = pd.concat([load.iloc[-1:], xgb_forecast])

# =========================
# 6. Plot
# =========================
plt.figure(figsize=(14, 7))

# 历史负荷（最后 200 小时）
plt.plot(
    load[-200:],
    label='Historical Load (Last 200 Hours)',
    color='blue',
    linewidth=2
)

# XGBoost 预测（无缝连接）
plt.plot(
    xgb_plot_series,
    label='XGBoost Forecast (Next 24 Hours)',
    color='green',
    linewidth=3
)

plt.title('XGBoost Load Forecasting (24-Hour Ahead)', fontsize=16)
plt.xlabel('Time')
plt.ylabel('Load (MW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
