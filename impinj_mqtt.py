import socket
import paho.mqtt.client as mqtt
import json
import joblib
import pandas as pd

# Impinj Speedway 连接配置
IMPINJ_HOST = "0.0.0.0"  # 监听所有 IP
IMPINJ_PORT = 14150       # Speedway Connect 发送数据的端口

# MQTT 服务器配置
MQTT_BROKER = "your-openhab-ip"
MQTT_RSSI_TOPIC = "rfid/data/rssi"
MQTT_PHASE_TOPIC = "rfid/data/phase"
MQTT_PREDICTION_TOPIC = "rfid/data/prediction"

# 加载训练好的模型
model = joblib.load("rfid_platform_classifier.pkl")

def predict_location(rssi, phase):
    data = pd.DataFrame({"RSSI": [rssi], "Phase": [phase]})
    predicted_location = model.predict(data)[0]
    print(f"Predicted Location: {predicted_location}")
    return predicted_location

# 连接 MQTT 服务器
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, 1883, 60)

# 创建 TCP 服务器监听 Impinj Speedway 数据
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((IMPINJ_HOST, IMPINJ_PORT))
server.listen(5)
print(f"Listening for Impinj Speedway data on {IMPINJ_PORT}...")

while True:
    client_socket, client_address = server.accept()
    print(f"Connected to {client_address}")

    while True:
        try:
            data = client_socket.recv(1024)
            if not data:
                break

            # 解析 Impinj Speedway JSON 数据
            rfid_data = json.loads(data.decode("utf-8"))
            tag_rssi = float(rfid_data.get("RSSI", -99))
            tag_phase = float(rfid_data.get("Phase", 0))

            print(f"Received: RSSI={tag_rssi}, Phase={tag_phase}")

            # 发送到 MQTT
            mqtt_client.publish(MQTT_RSSI_TOPIC, tag_rssi)
            mqtt_client.publish(MQTT_PHASE_TOPIC, tag_phase)

            # 进行机器学习推理
            predicted_location = predict_location(tag_rssi, tag_phase)
            mqtt_client.publish(MQTT_PREDICTION_TOPIC, predicted_location)

        except Exception as e:
            print(f"Error: {e}")
            break

    client_socket.close()
