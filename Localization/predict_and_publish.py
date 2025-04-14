import os
import time
import joblib
import paho.mqtt.publish as publish

FOLDER_PATH = r"C:\Users\Asus\Documents\HW2024-25\Project\DataSelect"
MODEL_PATH = "rfid_localizer.pkl"
MQTT_HOST = "192.168.2.198"
MQTT_PORT = 1883

epc_to_item = {
    "E280116060000207A639EC7A": "Item1",
    "E280116060000207A639F25A": "Item2",
}

model = joblib.load(MODEL_PATH)

last_seen = {}

def get_latest_file():
    files = [f for f in os.listdir(FOLDER_PATH) if f.startswith("LogTagData_") and f.endswith(".csv")]
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(FOLDER_PATH, f)), reverse=True)
    return os.path.join(FOLDER_PATH, files[0])

def tail_and_predict(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header_line = next((l for l in lines if "EPC" in l and l.startswith("//")), None)
    if not header_line:
        print("Header line not found.")
        return {}

    headers = [col.strip().upper() for col in header_line.replace("//", "").strip().split(',')]
    try:
        idx = {
            "EPC": headers.index("EPC"),
            "RSSI": headers.index("RSSI"),
            "PHASEANGLE": headers.index("PHASEANGLE"),
            "ANTENNA": headers.index("ANTENNA"),
        }
    except ValueError:
        print("Required column not found.")
        return {}

    data_lines = [l for l in lines if not l.startswith("//") and l.strip()]
    latest_lines = data_lines[-50:]

    results = {}
    for line in latest_lines:
        try:
            parts = line.strip().split(',')
            if len(parts) <= max(idx.values()):
                continue

            epc = parts[idx["EPC"]].strip().upper()
            if epc not in epc_to_item:
                continue

            rssi = float(parts[idx["RSSI"]].strip())
            phase = float(parts[idx["PHASEANGLE"]].strip())
            antenna = int(parts[idx["ANTENNA"]].strip())

            X = [[rssi, phase, antenna]]
            prediction = model.predict(X)[0]

            print(f"Tracked: {epc} -> Location={prediction}")
            results[epc] = prediction
        except Exception:
            continue
    return results

print("Monitoring latest CSV and predicting...")
try:
    latest_file = get_latest_file()
    print(f"Latest CSV: {latest_file}")
    while True:
        if latest_file and os.path.exists(latest_file):
            predictions = tail_and_predict(latest_file)
            now = time.time()

            for epc, position in predictions.items():
                item = epc_to_item[epc]

                location_topic = f"openhab/room/{item}/location"
                presence_topic = f"openhab/room/{item}/presence"

                publish.single(location_topic, payload=position, hostname=MQTT_HOST, port=MQTT_PORT)
                publish.single(presence_topic, payload="ON", hostname=MQTT_HOST, port=MQTT_PORT)

                print(f"[MQTT] {item} -> Location={position}, Presence=ON")
                last_seen[epc] = now

            for epc in list(last_seen):
                if now - last_seen[epc] > 5 and epc not in predictions:
                    item = epc_to_item[epc]
                    location_topic = f"openhab/room/{item}/location"
                    presence_topic = f"openhab/room/{item}/presence"

                    publish.single(presence_topic, payload="OFF", hostname=MQTT_HOST, port=MQTT_PORT)
                    publish.single(location_topic, payload="", hostname=MQTT_HOST, port=MQTT_PORT)

                    print(f"[MQTT] {item} -> Presence=OFF, Location=CLEARED")
                    del last_seen[epc]
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped by user.")
