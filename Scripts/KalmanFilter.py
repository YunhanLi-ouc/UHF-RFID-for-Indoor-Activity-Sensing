import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_path = "../Datasets/rfid_data_3p.csv"
rfid_data = pd.read_csv(file_path)

# Convert Timestamp to datetime format and sort data
rfid_data['Timestamp'] = pd.to_datetime(rfid_data['Timestamp'])
rfid_data = rfid_data.sort_values(by='Timestamp')

# Define Kalman Filter class
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

# Split data by platform
platforms = rfid_data['Platform_ID'].unique()
platform_data = {p: rfid_data[rfid_data['Platform_ID'] == p].copy() for p in platforms}

# Initialize Kalman Filters for each platform
kf_dict = {}
for p in platforms:
    measurement_variance_rssi = np.var(platform_data[p]['RSSI'].diff().dropna())  # RSSI noise variance
    measurement_variance_phase = np.var(platform_data[p]['Phase'].diff().dropna())  # Phase noise variance

    Q_rssi = 0.01 * measurement_variance_rssi  # Process variance for RSSI
    Q_phase = 0.01 * measurement_variance_phase  # Process variance for Phase

    kf_dict[p] = {
        'rssi': KalmanFilter(Q_rssi, measurement_variance_rssi, platform_data[p]['RSSI'].iloc[0]),
        'phase': KalmanFilter(Q_phase, measurement_variance_phase, platform_data[p]['Phase'].iloc[0])
    }

# Apply Kalman Filter to each platform separately
for p in platforms:
    platform_data[p]['RSSI_KF'] = [kf_dict[p]['rssi'].update(rssi) for rssi in platform_data[p]['RSSI']]
    platform_data[p]['Phase_KF'] = [kf_dict[p]['phase'].update(phase) for phase in platform_data[p]['Phase']]

# Standardize data within each platform
#for p in platforms:
    #platform_data[p]['RSSI_KF'] = (platform_data[p]['RSSI_KF'] - platform_data[p]['RSSI_KF'].mean()) / platform_data[p]['RSSI_KF'].std()
    #platform_data[p]['Phase_KF'] = (platform_data[p]['Phase_KF'] - platform_data[p]['Phase_KF'].mean()) / platform_data[p]['Phase_KF'].std()

# Merge the processed data
filtered_data = pd.concat(platform_data.values()).sort_values(by='Timestamp')
filtered_file_path = "../Datasets/rfid_data_filtered_per_platform.csv"
filtered_data.to_csv(filtered_file_path, index=False)

# Plot results
plt.figure(figsize=(10, 5))
for p in platforms:
    plt.plot(platform_data[p]['Timestamp'], platform_data[p]['RSSI'], label=f'Original RSSI - {p}', alpha=0.5)
    plt.plot(platform_data[p]['Timestamp'], platform_data[p]['RSSI_KF'], label=f'Filtered RSSI - {p}', linestyle='dashed')
plt.xlabel('Timestamp')
plt.ylabel('RSSI')
plt.legend()
plt.title('RSSI Filtering per Platform with Kalman Filter')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 5))
for p in platforms:
    plt.plot(platform_data[p]['Timestamp'], platform_data[p]['Phase'], label=f'Original Phase - {p}', alpha=0.5)
    plt.plot(platform_data[p]['Timestamp'], platform_data[p]['Phase_KF'], label=f'Filtered Phase - {p}', linestyle='dashed')
plt.xlabel('Timestamp')
plt.ylabel('Phase')
plt.legend()
plt.title('Phase Filtering per Platform with Kalman Filter')
plt.xticks(rotation=45)
plt.show()

print(f"Processed data saved to: {filtered_file_path}")
