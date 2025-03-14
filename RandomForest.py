import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
from collections import Counter

# Load data
file_path = "Datasets/rfid_data_3p.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Rename columns if Label represents Platform ID
df.rename(columns={"Label": "Platform_ID"}, inplace=True)

# Compute statistics for different platforms
platform_stats = df.groupby("Platform_ID")[["RSSI", "Phase"]].agg(["mean", "std", "min", "max"])
print("Platform statistics:")
print(platform_stats)

# Plot RSSI distribution
plt.figure(figsize=(12, 5))
sns.histplot(data=df, x="RSSI", hue="Platform_ID", kde=True, bins=30)
plt.title("RSSI Distribution (Different Platforms)")
plt.show()

# Plot Phase distribution
plt.figure(figsize=(12, 5))
sns.histplot(data=df, x="Phase", hue="Platform_ID", kde=True, bins=30)
plt.title("Phase Distribution (Different Platforms)")
plt.show()

# Scatter plot of RSSI vs Phase
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="RSSI", y="Phase", hue="Platform_ID", alpha=0.7)
plt.title("RSSI vs Phase Distribution (Different Platforms)")
plt.show()

# Compute Euclidean distance between platform means
platform_means = df.groupby("Platform_ID")[["RSSI", "Phase"]].mean()
distances = platform_means.apply(lambda row: euclidean(row, platform_means.iloc[0]), axis=1)
print("\nEuclidean distances between platforms (RSSI + Phase):")
print(distances)

# Compute correlation between RSSI and Phase
corr_matrix = df[["RSSI", "Phase"]].corr()
print("\nCorrelation Matrix (RSSI vs Phase):")
print(corr_matrix)

# Machine Learning Classification Model
# Extract features and labels
X = df[["RSSI", "Phase"]]
y = df["Platform_ID"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model
model_filename = "rfid_platform_classifier.pkl"
joblib.dump(clf, model_filename)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Compute and visualize confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# Real-time prediction function
def predict_platform(rssi, phase):
    clf_loaded = joblib.load("rfid_platform_classifier.pkl")
    new_data = pd.DataFrame({"RSSI": [rssi], "Phase": [phase]})
    predicted_platform = clf_loaded.predict(new_data)
    predicted_probabilities = clf_loaded.predict_proba(new_data)
    platforms = clf_loaded.classes_

    print(f"Predicted Platform: {predicted_platform[0]}")
    print("Prediction Probabilities:")
    for platform, prob in zip(platforms, predicted_probabilities[0]):
        print(f"Platform {platform}: {prob:.2%}")

    return predicted_platform[0]


# Example real-time prediction
real_time_rssi = -65.5
real_time_phase = 3.2
predicted = predict_platform(real_time_rssi, real_time_phase)

# Sliding window stabilization
recent_predictions = ["S1", "S1", "T1", "S1", "S1"]
most_common_platform = Counter(recent_predictions).most_common(1)[0][0]
print(f"Final Stable Prediction: {most_common_platform}")
