# des_tree.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler

DATASET_PATH = r"C:\Users\hp\Downloads\HAR VSC\UCI HAR Dataset"
OUTPUT_ROOT = os.path.join(os.path.dirname(DATASET_PATH), "outputs")
OUTPUTS_MODELS = os.path.join(OUTPUT_ROOT, "models")
OUTPUTS_FEATURES = os.path.join(OUTPUT_ROOT, "features")
os.makedirs(OUTPUTS_MODELS, exist_ok=True)

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")
INERTIAL_TRAIN = os.path.join(TRAIN_PATH, "Inertial Signals")
INERTIAL_TEST = os.path.join(TEST_PATH, "Inertial Signals")

#  Load inertial data
def load_inertial_split(path):
    X = {}
    subset = "train" if "train" in path else "test"
    for axis in ['x', 'y', 'z']:
        fname = os.path.join(path, f"total_acc_{axis}_{subset}.txt")
        X[axis] = pd.read_csv(fname, sep=r'\s+', header=None).values
    return X['x'], X['y'], X['z']

# Load Labels
train_labels = pd.read_csv(os.path.join(TRAIN_PATH, "y_train.txt"), header=None).values.ravel()
test_labels = pd.read_csv(os.path.join(TEST_PATH, "y_test.txt"), header=None).values.ravel()

# RAW ACCELEROMETER DATA

print("Loading raw accelerometer signals...")
train_x, train_y, train_z = load_inertial_split(INERTIAL_TRAIN)
test_x, test_y, test_z = load_inertial_split(INERTIAL_TEST)

train_raw = np.hstack([train_x, train_y, train_z])
test_raw = np.hstack([test_x, test_y, test_z])

scaler_raw = StandardScaler()
train_raw_scaled = scaler_raw.fit_transform(train_raw)
test_raw_scaled = scaler_raw.transform(test_raw)

print(f"Raw signals shape: Train {train_raw_scaled.shape}, Test {test_raw_scaled.shape}")


# PROVIDED FEATURE DATA
print("\nLoading provided engineered features...")
X_train = pd.read_csv(os.path.join(TRAIN_PATH, "X_train.txt"), sep=r'\s+', header=None)
X_test = pd.read_csv(os.path.join(TEST_PATH, "X_test.txt"), sep=r'\s+', header=None)

scaler_provided = StandardScaler()
X_train_scaled = scaler_provided.fit_transform(X_train)
X_test_scaled = scaler_provided.transform(X_test)

print(f"Provided features shape: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")


# TSFEL FEATURES
print("\nLoading TSFEL features...")
tsfel_train_path = os.path.join(OUTPUTS_FEATURES, "train_tsfel.csv")
tsfel_test_path = os.path.join(OUTPUTS_FEATURES, "test_tsfel.csv")

if os.path.exists(tsfel_train_path) and os.path.exists(tsfel_test_path):
    train_tsfel = pd.read_csv(tsfel_train_path)
    test_tsfel = pd.read_csv(tsfel_test_path)

    scaler_tsfel = StandardScaler()
    train_tsfel_scaled = scaler_tsfel.fit_transform(train_tsfel)
    test_tsfel_scaled = scaler_tsfel.transform(test_tsfel)

    print(f"TSFEL features shape: Train {train_tsfel_scaled.shape}, Test {test_tsfel_scaled.shape}")
    tsfel_available = True
else:
    print("TSFEL feature files not found. Skipping TSFEL comparison.")
    tsfel_available = False


# Function to evaluate and plot confusion matrix
def evaluate_model(model, X_train, y_train, X_test, y_test, label):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average='weighted')
    rec = recall_score(y_test, pred, average='weighted')
    print(f"\n[{label}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {label}")
    plt.savefig(os.path.join(OUTPUTS_MODELS, f"cm_{label}.png"))
    plt.close()
    return acc


# Train and Evaluate Models
print("\nTraining Decision Trees...")
acc_raw = evaluate_model(DecisionTreeClassifier(max_depth=6, random_state=42),
                         train_raw_scaled, train_labels, test_raw_scaled, test_labels, "RAW")
acc_provided = evaluate_model(DecisionTreeClassifier(max_depth=6, random_state=42),
                              X_train_scaled, train_labels, X_test_scaled, test_labels, "PROVIDED")
acc_tsfel = None
if tsfel_available:
    acc_tsfel = evaluate_model(DecisionTreeClassifier(max_depth=6, random_state=42),
                               train_tsfel_scaled, train_labels, test_tsfel_scaled, test_labels, "TSFEL")

# Accuracy vs Depth (2–8)
print("\nEvaluating depth from 2 to 8...")
depths = range(2, 9)
acc_curve = {"Raw": [], "Provided": [], "TSFEL": [] if tsfel_available else None}

for d in depths:
    acc_curve["Raw"].append(accuracy_score(test_labels,
        DecisionTreeClassifier(max_depth=d, random_state=42).fit(train_raw_scaled, train_labels).predict(test_raw_scaled)
    ))
    acc_curve["Provided"].append(accuracy_score(test_labels,
        DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train_scaled, train_labels).predict(X_test_scaled)
    ))
    if tsfel_available:
        acc_curve["TSFEL"].append(accuracy_score(test_labels,
            DecisionTreeClassifier(max_depth=d, random_state=42).fit(train_tsfel_scaled, train_labels).predict(test_tsfel_scaled)
        ))

plt.figure(figsize=(8,5))
plt.plot(depths, acc_curve["Raw"], label="Raw Signals", marker='o')
plt.plot(depths, acc_curve["Provided"], label="Provided Features", marker='o')
if tsfel_available:
    plt.plot(depths, acc_curve["TSFEL"], label="TSFEL Features", marker='o')
plt.xlabel("Tree Depth")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Tree Depth")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUTS_MODELS, "accuracy_vs_depth.png"))
plt.show()

print("\nTask 2 with TSFEL completed. Outputs saved in:", OUTPUTS_MODELS)
