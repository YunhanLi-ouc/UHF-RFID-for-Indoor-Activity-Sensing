import time
import os
import paho.mqtt.client as mqtt

FOLDER_PATH = r"C:\Users\Asus\Documents\HW2024-25\Project\DataSelect"
MQTT_BROKER = "192.168.2.198"
MQTT_PORT = 1883

# Mapping from EPC to MQTT topic
epc_to_topic = {
    # Kitchen
    "E28011606000000000000001": "kitchen/cabinet1",
    "E280116060000207A6391C6A": "kitchen/cabinet2",
    "E280116060000207A639C0A9": "kitchen/cabinet3",
    "E280116060000207A638AC6A": "kitchen/cabinet4",
    "4361116060000207A638E8EA": "kitchen/cabinet5",
    "E280116060000207A6391A4A": "kitchen/cabinet6",
    "E280116060000207A6391C5A": "kitchen/cabinet7",
    "E280116060000207A638AA3A": "kitchen/cabinet8",
    # Living Room
    "E280116060000207A639EC29": "livingroom/cabinet9",
    "E280116060000207A639ECDA": "livingroom/cabinet10"
}

last_seen = {}
seen_tags = set()

client = mqtt.Client(protocol=mqtt.MQTTv311)
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

def get_latest_file():
    files = [f for f in os.listdir(FOLDER_PATH) if f.startswith("LogTagData_") and f.endswith(".csv")]
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(FOLDER_PATH, f)), reverse=True)
    return os.path.join(FOLDER_PATH, files[0])

def tail_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        # 找表头（// Timestamp, EPC, ...）
        header_line = next((l for l in lines if "EPC" in l and l.startswith("//")), None)
        if header_line:
            headers = [col.strip() for col in header_line.replace("//", "").strip().split(',')]
        else:
            print("Cannot find column title line containing 'EPC'")
            return []

        epc_index = next((i for i, col in enumerate(headers) if col.strip().upper() == "EPC"), -1)
        if epc_index == -1:
            print("Unrecognized 'EPC' column“")
            return []

        data_lines = [l for l in lines if not l.strip().startswith("//") and l.strip()]
        latest_lines = data_lines[-30:]

        epcs = set()
        for line in latest_lines:
            try:
                fields = line.strip().split(',')
                if len(fields) > epc_index:
                    epc = fields[epc_index].strip().upper()
                    epcs.add(epc)
            except Exception:
                continue
        return epcs

def update_tags(epcs):
    global seen_tags, last_seen
    now = time.time()

    for epc in epcs:
        if epc not in seen_tags:
            topic = epc_to_topic.get(epc)
            if topic:
                print(f"[+] Appear: {topic} ({epc})")
                client.publish(f"uhf/{topic}", "ON")
        seen_tags.add(epc)
        last_seen[epc] = now

    # No timeout detected during inspection
    to_remove = []
    for epc in list(last_seen):
        if now - last_seen[epc] > 5 and epc not in epcs:
            topic = epc_to_topic.get(epc)
            if topic:
                print(f"[-] Disappear: {topic} ({epc})")
                client.publish(f"uhf/{topic}", "OFF")
            to_remove.append(epc)
            seen_tags.discard(epc)
    for epc in to_remove:
        del last_seen[epc]

print("Monitoring...")
try:
    latest_file = get_latest_file()
    print(f"📄 logfile: {latest_file}")
    while True:
        if latest_file and os.path.exists(latest_file):
            current_epcs = tail_csv(latest_file)
            update_tags(current_epcs)
        time.sleep(1)
except KeyboardInterrupt:
    print("Stop Monitoring.")
    client.loop_stop()
