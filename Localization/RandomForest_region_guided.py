import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# === Step 1: 加载数据 ===
file_path = "../Datasets/rfid_data_filtered_per_platform.csv"
df = pd.read_csv(file_path)

# 确保时间顺序
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values(by='Timestamp')

# 统一标签列名
df.rename(columns={"Label": "Platform_ID"}, inplace=True)

# === Step 2: 特征选择 ===
# 仅使用 Antenna_ID、RSSI_KF、Phase_KF 作为输入特征
features = ["Antenna_ID", "RSSI_KF", "Phase_KF"]
X = df[features]
y = df["Platform_ID"]

# === Step 3: 数据划分 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 4: 模型训练 ===
clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# === Step 5: 模型保存 ===
model_filename = "rfid_localizer.pkl"
joblib.dump(clf, model_filename)
print(f"The model has been saved: {model_filename}")

# === Step 6: 性能评估 ===
y_pred = clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel("Prediction label")
plt.ylabel("True Label")
plt.title("Complex Matrix")
plt.show()

# 特征重要性
importance = clf.feature_importances_
plt.figure(figsize=(6, 4))
plt.barh(features, importance)
plt.xlabel("Importance of Features")
plt.title("Feature contribution analysis")
plt.show()
