import time
import os
import glob
import pandas as pd
import paho.mqtt.client as mqtt
from joblib import load

# --- Configuration ---
LOG_DIR = r"C:\\Users\\Asus\\Documents\\HW2024-25\\Project\\DataSelect"
MODEL_FILE = "../rfid_platform_classifier.pkl"
MQTT_BROKER = "localhost"
TOPIC_RSSI = "rfid/data/rssi"
TOPIC_PHASE = "rfid/data/phase"
TOPIC_PREDICTION = "rfid/data/prediction"

# --- MQTT Setup ---
client = mqtt.Client()
client.connect(MQTT_BROKER)
client.loop_start()

# --- Load Model ---
model = load(MODEL_FILE)

def predict_location(rssi, phase):
    df = pd.DataFrame([[rssi, phase]], columns=["RSSI", "Phase"])
    pred = model.predict(df)[0]
    print(f"Predicted: {pred} from RSSI={rssi}, Phase={phase}")
    client.publish(TOPIC_PREDICTION, str(pred))

def get_latest_log_file(log_dir):
    csv_files = glob.glob(os.path.join(log_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in log directory.")
    latest = max(csv_files, key=os.path.getctime)
    print(f"📄 最新日志文件：{latest}")
    return latest

def tail_and_process(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            if 'EPC' in line or not line.strip():
                continue
            try:
                parts = line.strip().split(',')
                rssi = float(parts[3])    # Adjust index based on actual CSV
                phase = float(parts[4])
                client.publish(TOPIC_RSSI, rssi)
                client.publish(TOPIC_PHASE, phase)
                predict_location(rssi, phase)
            except Exception as e:
                print(f"Failed to parse line: {line.strip()}. Error: {e}")

if __name__ == "__main__":
    log_file = get_latest_log_file(LOG_DIR)
    print(f"Monitoring RFID log: {log_file}")
    tail_and_process(log_file)
