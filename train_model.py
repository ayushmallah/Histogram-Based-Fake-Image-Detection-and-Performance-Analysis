import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

REAL_PATH = "dataset/real"
FAKE_PATH = "dataset/fake"

# 🔹 Feature Extraction (UPDATED & STABLE)
def extractImageData(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.resize(image, (256, 256))

    # RGB Histograms
    hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])

    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean = np.mean(gray)
    std = np.std(gray)

    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    gaps = np.sum(hist_gray == 0) / 256   # 🔴 FIXED (normalized)

    # Edge
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (256 * 256)

    # Noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray - blur
    noise_level = np.std(noise)

    features = np.hstack([
        hist_r, hist_g, hist_b,
        mean, std, gaps,
        edge_density, noise_level
    ])

    return np.array(features)


# 🔹 Load dataset
X, y = [], []
valid_ext = (".png", ".jpg", ".jpeg")


# Real = 1
for file in os.listdir(REAL_PATH):
    if not file.lower().endswith(valid_ext):
        continue
    path = os.path.join(REAL_PATH, file)
    features = extractImageData(path)
    if features is not None:
        X.append(features)
        y.append(1)

# Fake = 0
for file in os.listdir(FAKE_PATH):
    if not file.lower().endswith(valid_ext):
        continue
    path = os.path.join(FAKE_PATH, file)
    features = extractImageData(path)
    if features is not None:
        X.append(features)
        y.append(0)

X = np.array(X)
y = np.array(y)

# 🔴 Safety check
if len(X) == 0:
    exit()


# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 🔹 Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# 🔹 Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# 🔹 Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)


# 🔹 Save model
joblib.dump(model, "model.pkl")


# 🔴 Save directory
save_dir = os.getcwd()


# =========================
# 🔹 ROC Curve
# =========================
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "roc_curve.png"))
plt.close()


# =========================
# 🔹 Confusion Matrix
# =========================
plt.figure()
sns.heatmap(cm, annot=True, fmt="d")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
plt.close()
