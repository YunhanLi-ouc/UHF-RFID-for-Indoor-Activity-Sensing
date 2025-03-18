import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load Kalman-filtered dataset
file_path = "Datasets/rfid_data_filtered.csv"
df = pd.read_csv(file_path)

# Ensure proper time order
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values(by='Timestamp')

# Rename label column if necessary
df.rename(columns={"Label": "Platform_ID"}, inplace=True)

# Select Features and Labels
features = ["RSSI_KF", "Phase_KF", "Antenna_ID"]
X = df[features]
y = df["Platform_ID"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model
model_filename = "rfid_platform_classifier.pkl"
joblib.dump(clf, model_filename)
print(f"Trained model saved as: {model_filename}")

# Evaluate Model Performance
y_pred = clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance Analysis
importances = clf.feature_importances_
feature_names = features
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Classification")
plt.show()
