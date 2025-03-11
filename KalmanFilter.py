import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
file_path = "rfid_data_t2.csv"  # 请替换为你的数据文件路径
rfid_data = pd.read_csv(file_path)

# 确保时间戳格式正确
rfid_data['Timestamp'] = pd.to_datetime(rfid_data['Timestamp'])
rfid_data = rfid_data.sort_values(by='Timestamp')

# 定义卡尔曼滤波器类
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_estimate=0, initial_error=1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_estimate
        self.error = initial_error

    def update(self, measurement):
        # 预测更新
        self.error += self.process_variance

        # 观测更新（修正）
        kalman_gain = self.error / (self.error + self.measurement_variance)
        self.estimate += kalman_gain * (measurement - self.estimate)
        self.error = (1 - kalman_gain) * self.error

        return self.estimate

# 估算测量噪声
process_variance = 1e-2  # 设定一个小的过程噪声
measurement_variance_rssi = np.var(rfid_data['RSSI'].diff().dropna())  # 计算 RSSI 变化的方差
measurement_variance_phase = np.var(rfid_data['Phase'].diff().dropna())  # 计算 Phase 变化的方差

# 初始化卡尔曼滤波器
kf_rssi = KalmanFilter(process_variance, measurement_variance_rssi, initial_estimate=rfid_data['RSSI'].iloc[0])
kf_phase = KalmanFilter(process_variance, measurement_variance_phase, initial_estimate=rfid_data['Phase'].iloc[0])

# 应用卡尔曼滤波器
rfid_data['RSSI_KF'] = [kf_rssi.update(rssi) for rssi in rfid_data['RSSI']]
rfid_data['Phase_KF'] = [kf_phase.update(phase) for phase in rfid_data['Phase']]

# 保存处理后的数据
filtered_file_path = "rfid_data_filtered.csv"  # 你可以更改文件路径
rfid_data.to_csv(filtered_file_path, index=False)

# 绘制 RSSI 滤波前后对比
plt.figure(figsize=(10, 5))
plt.plot(rfid_data['Timestamp'], rfid_data['RSSI'], label='Original RSSI', alpha=0.5)
plt.plot(rfid_data['Timestamp'], rfid_data['RSSI_KF'], label='Kalman Filtered RSSI', linestyle='dashed')
plt.xlabel('Timestamp')
plt.ylabel('RSSI')
plt.legend()
plt.title('RSSI Filtering with Kalman Filter')
plt.xticks(rotation=45)
plt.show()

# 绘制 Phase 滤波前后对比
plt.figure(figsize=(10, 5))
plt.plot(rfid_data['Timestamp'], rfid_data['Phase'], label='Original Phase', alpha=0.5)
plt.plot(rfid_data['Timestamp'], rfid_data['Phase_KF'], label='Kalman Filtered Phase', linestyle='dashed')
plt.xlabel('Timestamp')
plt.ylabel('Phase')
plt.legend()
plt.title('Phase Filtering with Kalman Filter')
plt.xticks(rotation=45)
plt.show()

print(f"处理后的数据已保存至: {filtered_file_path}")
