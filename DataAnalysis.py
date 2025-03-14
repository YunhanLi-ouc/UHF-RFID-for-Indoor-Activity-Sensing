import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean

# 读取数据
file_path = "Datasets/rfid_data_3p.csv"  # 请替换为你的文件路径
df = pd.read_csv(file_path)

# 查看数据基本信息
print("数据概览:")
print(df.head())

# 统计各个平台的 RSSI 和 Phase 的均值、标准差、最大最小值
platform_stats = df.groupby("Platform_ID")[["RSSI", "Phase"]].agg(["mean", "std", "min", "max"])
print("\n不同平台的统计特性:")
print(platform_stats)

# 可视化 RSSI 分布
plt.figure(figsize=(12, 5))
sns.histplot(data=df, x="RSSI", hue="Platform_ID", kde=True, bins=30)
plt.title("RSSI 分布 (不同平台)")
plt.show()

# 可视化 Phase 分布
plt.figure(figsize=(12, 5))
sns.histplot(data=df, x="Phase", hue="Platform_ID", kde=True, bins=30)
plt.title("Phase 分布 (不同平台)")
plt.show()

# RSSI 与 Phase 散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="RSSI", y="Phase", hue="Platform_ID", alpha=0.7)
plt.title("RSSI vs Phase 分布 (不同平台)")
plt.show()

# 计算平台间的欧几里得距离
platform_means = df.groupby("Platform_ID")[["RSSI", "Phase"]].mean()
distances = platform_means.apply(lambda row: euclidean(row, platform_means.iloc[0]), axis=1)
print("\n不同平台的 RSSI + Phase 欧几里得距离:")
print(distances)

# 计算 RSSI 和 Phase 之间的相关性
corr_matrix = df[["RSSI", "Phase"]].corr()
print("\nRSSI 与 Phase 相关性矩阵:")
print(corr_matrix)