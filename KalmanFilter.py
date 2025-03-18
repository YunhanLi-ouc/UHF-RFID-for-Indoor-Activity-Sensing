import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
file_path = "Datasets/rfid_data_3p.csv"
rfid_data = pd.read_csv(file_path)

rfid_data['Timestamp'] = pd.to_datetime(rfid_data['Timestamp'])
rfid_data = rfid_data.sort_values(by='Timestamp')

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_estimate=0, initial_error=1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_estimate
        self.error = initial_error

    def update(self, measurement):
        # Prediction update
        self.error += self.process_variance

        # Observation update
        kalman_gain = self.error / (self.error + self.measurement_variance)
        self.estimate += kalman_gain * (measurement - self.estimate)
        self.error = (1 - kalman_gain) * self.error

        return self.estimate

# Estimate measurement noise
measurement_variance_rssi = np.var(rfid_data['RSSI'].diff().dropna())  # Calculate the variance of RSSI variation
measurement_variance_phase = np.var(rfid_data['Phase'].diff().dropna())  # Calculate the variance of Phase variation

Q_rssi = 0.01 * measurement_variance_rssi
Q_phase = 0.01 * measurement_variance_phase

# Initialize Kalman filter
kf_rssi = KalmanFilter(Q_rssi, measurement_variance_rssi, initial_estimate=rfid_data['RSSI'].iloc[0])
kf_phase = KalmanFilter(Q_phase, measurement_variance_phase, initial_estimate=rfid_data['Phase'].iloc[0])

rfid_data['RSSI_KF'] = [kf_rssi.update(rssi) for rssi in rfid_data['RSSI']]
rfid_data['Phase_KF'] = [kf_phase.update(phase) for phase in rfid_data['Phase']]

filtered_file_path = "Datasets/rfid_data_filtered.csv" #Save the filtered dataset
rfid_data.to_csv(filtered_file_path, index=False)

# Draw the image
plt.figure(figsize=(10, 5))
plt.plot(rfid_data['Timestamp'], rfid_data['RSSI'], label='Original RSSI', alpha=0.5)
plt.plot(rfid_data['Timestamp'], rfid_data['RSSI_KF'], label='Kalman Filtered RSSI', linestyle='dashed')
plt.xlabel('Timestamp')
plt.ylabel('RSSI')
plt.legend()
plt.title('RSSI Filtering with Kalman Filter')
plt.xticks(rotation=45)
plt.show()

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
