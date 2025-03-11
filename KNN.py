import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1️⃣  加载数据集（请替换 'your_data.csv' 为实际文件名）
df = pd.read_csv('rfid_data_timestamp.csv')  # 确保数据文件包含 RSSI, Phase, Label

# 2️⃣  选择特征（仅使用 RSSI 和 Phase）
X = df[['RSSI', 'Phase']].values  # 只选取 RSSI 和 Phase 作为特征
y = df['Label'].values  # 位置标签（T1, T2, T3, ...）

# 3️⃣  标签编码（将 T1, T2, ... 转换为 0, 1, 2, ...）
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4️⃣  数据归一化（标准化 RSSI 和 Phase）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5️⃣  数据集划分（训练集 80%，测试集 20%）
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 6️⃣  训练 KNN 模型（选择 k=5，适合一般情况，可调节）
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 7️⃣  预测测试集
y_pred = knn.predict(X_test)

# 8️⃣  计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 9️⃣  绘制混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("KNN Confusion Matrix (RSSI & Phase)")
plt.show()

# 🔟  分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 11️⃣  绘制分类结果的散点图
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='Set1', alpha=0.7, edgecolors='k')
plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)), label="Predicted Label")
plt.xticks([])
plt.yticks([])
plt.xlabel("Normalized RSSI")
plt.ylabel("Normalized Phase")
plt.title("KNN Test Set Classification Results (Using RSSI & Phase Only)")
plt.show()
