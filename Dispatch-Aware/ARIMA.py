# =========================
# 1. Import libraries
# =========================
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# =========================
# 2. Load data
# =========================
file_path = 'load_data.csv'

data = pd.read_csv(
    file_path,
    parse_dates=['Datetime'],
    index_col='Datetime'
).sort_index()

# 选择负荷列（根据你的数据集改成 'AEP_MW' 或 'PJME_MW'）
load = data['AEP_MW']

# =========================
# 关键修复：处理重复索引 + 设置频率
# =========================
# 步骤1：如果有重复时间戳，取平均值（最安全的聚合方式）
if load.index.duplicated().any():
    print("检测到重复时间索引，正在聚合取平均值...")
    load = load.groupby(load.index).mean()

# 步骤2：设置每小时频率（小写 'h'，新版推荐）
load = load.asfreq('h')

# =========================
# 3. Train ARIMA model
# =========================
train_data = load[-5000:]

print("Training ARIMA model...")
model = ARIMA(train_data, order=(5, 1, 0))
model_fit = model.fit()

# =========================
# 4. Forecast next 24 hours（最推荐方式）
# =========================
forecast_steps = 24
forecast_result = model_fit.get_forecast(steps=forecast_steps)
forecast_values = forecast_result.predicted_mean
forecast_index = forecast_result.predicted_mean.index  # 自动带频率的时间索引

forecast = pd.Series(forecast_values.values, index=forecast_index)

# =========================
# 5. 让预测线完美连接历史最后一个点
# =========================
arima_plot_series = pd.concat([load.iloc[-1:], forecast])

# =========================
# 6. Plot results
# =========================
plt.figure(figsize=(12, 6))

# 历史负荷（最后 200 小时）
plt.plot(
    train_data[-200:],
    label='Historical Load (Last 200 Hours)',
    linewidth=2,
    color='blue'
)

# 预测部分（自然衔接）
plt.plot(
    arima_plot_series,
    label='ARIMA Forecast (Next 24 Hours)',
    linewidth=3,
    color='red'
)

plt.title('Electric Load Forecasting using ARIMA')
plt.xlabel('Time')
plt.ylabel('Load (MW)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
