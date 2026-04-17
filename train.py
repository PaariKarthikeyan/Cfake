"""
CFAKE: Real vs AI-Generated Images
Train Logistic Regression, Random Forest, Decision Tree
Dataset: https://www.kaggle.com/datasets/birdy654/CIFAKE-real-and-ai-generated-synthetic-images
"""

import os
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_PATH = "dataset"       # root folder with train/ and test/ sub-folders
IMG_SIZE     = (32, 32)
MAX_TRAIN    = 10000           # samples per class; increase to 50000 for full accuracy
MAX_TEST     = 2000
MODEL_DIR    = "models"
STATIC_DIR   = "static"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(f"{STATIC_DIR}/charts", exist_ok=True)

# ─── DATA LOADER ──────────────────────────────────────────────────────────────
def load_images(folder, label, max_samples):
    images, labels = [], []
    for fname in os.listdir(folder)[:max_samples]:
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path).convert("RGB").resize(IMG_SIZE)
            images.append(np.array(img).flatten() / 255.0)
            labels.append(label)
        except Exception:
            pass
    return images, labels

def load_split(split="train", max_per_class=MAX_TRAIN):
    base     = os.path.join(DATASET_PATH, split)
    real_dir = os.path.join(base, "REAL")
    fake_dir = os.path.join(base, "FAKE")
    X, y = [], []
    for d, lbl in [(real_dir, 0), (fake_dir, 1)]:
        imgs, lbls = load_images(d, lbl, max_per_class)
        X.extend(imgs)
        y.extend(lbls)

    # ✅ Shuffle to prevent order bias
    combined = list(zip(X, y))
    np.random.seed(42)
    np.random.shuffle(combined)
    X, y = zip(*combined)
    return np.array(X), np.array(y)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
print("Loading training data...")
X_train, y_train = load_split("train", MAX_TRAIN)
print(f"  Train → {X_train.shape}  | REAL={sum(y_train==0)}  FAKE={sum(y_train==1)}")

print("Loading test data...")
X_test, y_test = load_split("test", MAX_TEST)
print(f"  Test  → {X_test.shape}  | REAL={sum(y_test==0)}  FAKE={sum(y_test==1)}")

# ─── PREPROCESSING: SCALE + PCA ───────────────────────────────────────────────
print("\nApplying StandardScaler + PCA (150 components)...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

pca = PCA(n_components=150, random_state=42)
X_train_p = pca.fit_transform(X_train_s)
X_test_p  = pca.transform(X_test_s)
print(f"  Explained variance (150 PCs): {pca.explained_variance_ratio_.sum():.3f}")

# ─── MODELS ───────────────────────────────────────────────────────────────────
# ✅ Forces models to weight both classes equally regardless of load order
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0,
                                               class_weight='balanced',  # ADD THIS
                                               random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=15,
                                                   class_weight='balanced',  # ADD THIS
                                                   random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=20,
                                                   class_weight='balanced',  # ADD THIS
                                                   n_jobs=-1, random_state=42),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    t0 = time.time()
    model.fit(X_train_p, y_train)
    train_time = round(time.time() - t0, 2)

    y_pred  = model.predict(X_test_p)
    y_proba = model.predict_proba(X_test_p)[:, 1]

    acc  = round(accuracy_score(y_test, y_pred)              * 100, 2)
    prec = round(precision_score(y_test, y_pred)             * 100, 2)
    rec  = round(recall_score(y_test, y_pred)                * 100, 2)
    f1   = round(f1_score(y_test, y_pred)                    * 100, 2)
    auc  = round(roc_auc_score(y_test, y_proba)              * 100, 2)
    cm   = confusion_matrix(y_test, y_pred).tolist()

    results[name] = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": auc, "train_time": train_time,
        "confusion_matrix": cm
    }
    print(f"  Accuracy={acc}%  F1={f1}%  AUC={auc}%  Time={train_time}s")

# ─── BEST MODEL ───────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["f1"])
results["best_model"] = best_name
print(f"\n✅ Best Model: {best_name}  (F1={results[best_name]['f1']}%)")

# ─── SAVE MODELS & RESULTS ────────────────────────────────────────────────────
with open(f"{MODEL_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

for name, model in models.items():
    safe_name = name.replace(" ", "_").lower()
    with open(f"{MODEL_DIR}/{safe_name}.pkl", "wb") as f:
        pickle.dump(model, f)

with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(f"{MODEL_DIR}/pca.pkl", "wb") as f:
    pickle.dump(pca, f)

print("\nAll models and artifacts saved to /models")

# ─── CHART COLORS ─────────────────────────────────────────────────────────────
colors = {
    "Logistic Regression": "#6366f1",
    "Decision Tree":       "#f59e0b",
    "Random Forest":       "#10b981"
}
metrics = ["accuracy", "precision", "recall", "f1", "auc"]

# ─── CHART 1: Metrics Comparison Bar Chart ────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(metrics))
w = 0.25

for i, name in enumerate(models):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i * w, vals, w, label=name, color=colors[name], alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                str(v), ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x + w)
ax.set_xticklabels([m.upper() for m in metrics])
ax.set_ylim(50, 107)
ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{STATIC_DIR}/charts/metrics_comparison.png", dpi=150)
plt.close()
print("Chart 1 saved: metrics_comparison.png")

# ─── CHART 2: Confusion Matrices ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, name in zip(axes, models):
    cm = np.array(results[name]["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"],
                linewidths=0.5, linecolor="gray")
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{STATIC_DIR}/charts/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 2 saved: confusion_matrices.png")

# ─── CHART 3: Training Time ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
names = list(models.keys())
times = [results[n]["train_time"] for n in names]
bars  = ax.bar(names, times, color=[colors[n] for n in names], alpha=0.9, width=0.5)
for b, t in zip(bars, times):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
            f"{t}s", ha="center", fontsize=10)
ax.set_ylabel("Seconds")
ax.set_title("Training Time per Model", fontsize=14, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{STATIC_DIR}/charts/training_time.png", dpi=150)
plt.close()
print("Chart 3 saved: training_time.png")

# ─── CHART 4: Radar Chart ─────────────────────────────────────────────────────
radar_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
N      = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for name in models:
    vals = [results[name][m] / 100 for m in metrics]
    vals += vals[:1]
    ax.plot(angles, vals, label=name, color=colors[name], linewidth=2)
    ax.fill(angles, vals, color=colors[name], alpha=0.1)

ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_title("Radar: Model Metrics", fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.1))
plt.tight_layout()
plt.savefig(f"{STATIC_DIR}/charts/radar_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 4 saved: radar_chart.png")

# ─── CHART 5: PCA Explained Variance ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
ax.plot(range(1, len(cumvar) + 1), cumvar, color="#6366f1", linewidth=2)
ax.axhline(y=cumvar[-1], color="#f59e0b", linestyle="--",
           label=f"{cumvar[-1]:.1f}% total variance")
ax.fill_between(range(1, len(cumvar) + 1), cumvar, alpha=0.15, color="#6366f1")
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Explained Variance (%)")
ax.set_title("PCA Explained Variance", fontsize=14, fontweight="bold")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{STATIC_DIR}/charts/pca_variance.png", dpi=150)
plt.close()
print("Chart 5 saved: pca_variance.png")

print("\n=== Training complete! Run app.py to start the server. ===")