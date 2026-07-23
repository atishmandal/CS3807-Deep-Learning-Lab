import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import Perceptron as SklearnPerceptron
import os


PLOT_DIR = "plots"
TABLE_DIR = "tables"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)


print("=" * 70)
print("TASK 1: DATASET EXPLORATION")
print("=" * 70)

columns = ["Variance", "Skewness", "Curtosis", "Entropy", "Class"]

PRIMARY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
FALLBACK_URL = "https://raw.githubusercontent.com/Kuntal-G/Machine-Learning/master/R-machine-learning/data/banknote-authentication.csv"

try:
    data = pd.read_csv(PRIMARY_URL, names=columns)
except Exception as e:
    print(f"Primary UCI URL failed ({e}); falling back to mirrored dataset...")
    data = pd.read_csv(FALLBACK_URL)
    data.columns = columns

print(f"\nDataset Shape: {data.shape}")
print("\nFirst 5 Rows:")
print(data.head())
print("\nMissing Values:\n", data.isnull().sum())
print("\nDescriptive Statistics:\n", data.describe())
print("\nClass Balance:\n", data["Class"].value_counts())

print("\n" + "=" * 70)
print("TASK 2: EXPLORATORY DATA ANALYSIS")
print("=" * 70)


fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.flatten(), columns[:-1]):
    sns.histplot(data=data, x=col, hue="Class", kde=True, ax=ax, palette=["#2E86AB", "#E63946"])
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_feature_histograms.png", dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/01_feature_histograms.png")


plt.figure(figsize=(6, 5))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_correlation_heatmap.png", dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/02_correlation_heatmap.png")


plt.figure(figsize=(6, 5))
sns.scatterplot(data=data, x="Variance", y="Skewness", hue="Class",
                 palette=["#2E86AB", "#E63946"], alpha=0.6)
plt.title("Variance vs Skewness (colored by Class)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_scatter_variance_skewness.png", dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/03_scatter_variance_skewness.png")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.flatten(), columns[:-1]):
    sns.boxplot(data=data, x="Class", y=col, ax=ax, palette=["#2E86AB", "#E63946"])
    ax.set_title(f"Boxplot of {col} by Class")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_boxplots.png", dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/04_boxplots.png")


print("\n" + "=" * 70)
print("TASK 3: DATA PREPROCESSING")
print("=" * 70)

X = data.drop(columns=["Class"]).values
y = data["Class"].values


X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X - X_min) / (X_max - X_min)


X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")



class PerceptronFromScratch:
    """Single Layer Perceptron trained with the classical perceptron
    learning rule and a hard Step Activation Function."""

    def __init__(self, learning_rate=0.01, epochs=50):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

        self.history_errors = []
        self.history_weights = []
        self.history_bias = []
        self.converged_epoch = None

    def _step_activation(self, z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y, verbose=True):
        num_features = X.shape[1]
        self.weights = np.zeros(num_features)
        self.bias = 0.0

        for epoch in range(1, self.epochs + 1):
            errors_in_epoch = 0

            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                prediction = self._step_activation(linear_output)

                update = self.lr * (y[idx] - prediction)
                if update != 0:
                    self.weights += update * x_i
                    self.bias += update
                    errors_in_epoch += 1

            self.history_errors.append(errors_in_epoch)
            self.history_weights.append(self.weights.copy())
            self.history_bias.append(self.bias)

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"Epoch {epoch:3d} | Misclassified: {errors_in_epoch:3d} | "
                      f"Weights: {np.round(self.weights, 4)} | Bias: {round(self.bias, 4)}")

            if errors_in_epoch == 0:
                self.converged_epoch = epoch
                if verbose:
                    print(f"--> Converged early at epoch {epoch}!")
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._step_activation(linear_output)


print("\n" + "=" * 70)
print("TASK 5: MODEL TRAINING (learning_rate = 0.01)")
print("=" * 70)

model = PerceptronFromScratch(learning_rate=0.01, epochs=40)
model.fit(X_train, y_train)


print("\n" + "=" * 70)
print("TASK 6: MODEL EVALUATION")
print("=" * 70)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")

# --- Training Error vs Epoch ---
epochs_range = range(1, len(model.history_errors) + 1)
plt.figure(figsize=(6, 4.5))
plt.plot(epochs_range, model.history_errors, marker='o', color='red')
plt.title("Training Error vs Epoch (lr=0.01)")
plt.xlabel("Epoch")
plt.ylabel("Misclassified Count")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_training_error_vs_epoch.png", dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/05_training_error_vs_epoch.png")

# --- Weight Evolution ---
plt.figure(figsize=(6, 4.5))
plt.plot(epochs_range, model.history_weights)
plt.title("Weight Evolution (lr=0.01)")
plt.xlabel("Epoch")
plt.ylabel("Weight Value")
plt.legend([f"W_{c}" for c in columns[:-1]])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/06_weight_evolution.png", dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/06_weight_evolution.png")

# --- Bias Evolution ---
plt.figure(figsize=(6, 4.5))
plt.plot(epochs_range, model.history_bias, color='purple')
plt.title("Bias Evolution (lr=0.01)")
plt.xlabel("Epoch")
plt.ylabel("Bias Value")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/07_bias_evolution.png", dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/07_bias_evolution.png")

# --- Confusion Matrix Heatmap ---
plt.figure(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Authentic (0)', 'Forged (1)'],
            yticklabels=['Authentic (0)', 'Forged (1)'])
plt.title("Confusion Matrix (lr=0.01)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/08_confusion_matrix.png", dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/08_confusion_matrix.png")


#Learning rate comparision
print("\n" + "=" * 70)
print("TASK 7: LEARNING RATE COMPARISON (0.001, 0.01, 0.1)")
print("=" * 70)

learning_rates = [0.001, 0.01, 0.1]
lr_results = {}

for lr in learning_rates:
    print(f"\n--- Training with learning_rate = {lr} ---")
    m = PerceptronFromScratch(learning_rate=lr, epochs=40)
    m.fit(X_train, y_train, verbose=False)
    preds = m.predict(X_test)

    lr_results[lr] = {
        "model": m,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "epochs_to_converge": m.converged_epoch if m.converged_epoch else m.epochs,
        "converged": m.converged_epoch is not None,
        "final_errors": m.history_errors[-1],
    }
    if lr_results[lr]["converged"]:
        print(f"Converged at epoch: {lr_results[lr]['epochs_to_converge']} "
              f"(final epoch misclassified: {lr_results[lr]['final_errors']})")
    else:
        print(f"Did NOT fully converge within {m.epochs} epochs "
              f"(final epoch misclassified: {lr_results[lr]['final_errors']} / "
              f"{X_train.shape[0]} training samples). This is expected since the "
              f"banknote dataset is not perfectly linearly separable.")
    print(f"Accuracy: {lr_results[lr]['accuracy']:.4f} | "
          f"Precision: {lr_results[lr]['precision']:.4f} | "
          f"Recall: {lr_results[lr]['recall']:.4f} | "
          f"F1: {lr_results[lr]['f1']:.4f}")
# Learning Rate Comparison Plot ---
plt.figure(figsize=(7, 5))
for lr in learning_rates:
    hist = lr_results[lr]["model"].history_errors
    plt.plot(range(1, len(hist) + 1), hist, marker='o', markersize=3, label=f"lr={lr}")
plt.title("Training Error vs Epoch for Different Learning Rates")
plt.xlabel("Epoch")
plt.ylabel("Misclassified Count")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/09_learning_rate_comparison.png", dpi=150)
plt.close()
print(f"\nSaved: {PLOT_DIR}/09_learning_rate_comparison.png")

# Learning Rate Summary Table ---
lr_summary = pd.DataFrame({
    "Learning Rate": learning_rates,
    "Epochs Run": [lr_results[lr]["epochs_to_converge"] for lr in learning_rates],
    "Fully Converged": [lr_results[lr]["converged"] for lr in learning_rates],
    "Accuracy": [round(lr_results[lr]["accuracy"], 4) for lr in learning_rates],
    "Precision": [round(lr_results[lr]["precision"], 4) for lr in learning_rates],
    "Recall": [round(lr_results[lr]["recall"], 4) for lr in learning_rates],
    "F1-score": [round(lr_results[lr]["f1"], 4) for lr in learning_rates],
})
print("\nLearning Rate Summary Table:")
print(lr_summary.to_string(index=False))

lr_summary.to_csv(f"{TABLE_DIR}/lr_comparison_summary.csv", index=False)
print(f"Saved: {TABLE_DIR}/lr_comparison_summary.csv")

# Weight Evolution across Learning Rates (one subplot per feature) 
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for feat_idx, (ax, col) in enumerate(zip(axes.flatten(), columns[:-1])):
    for lr in learning_rates:
        w_hist = lr_results[lr]["model"].history_weights
        w_series = [w[feat_idx] for w in w_hist]
        ax.plot(range(1, len(w_series) + 1), w_series, marker='o', markersize=3, label=f"lr={lr}")
    ax.set_title(f"Weight Evolution: {col}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight Value")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/11_weight_evolution_by_lr.png", dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/11_weight_evolution_by_lr.png")

# Bias Evolution across Learning Rates (overlaid) 
plt.figure(figsize=(7, 5))
for lr in learning_rates:
    b_hist = lr_results[lr]["model"].history_bias
    plt.plot(range(1, len(b_hist) + 1), b_hist, marker='o', markersize=3, label=f"lr={lr}")
plt.title("Bias Evolution for Different Learning Rates")
plt.xlabel("Epoch")
plt.ylabel("Bias Value")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/12_bias_evolution_by_lr.png", dpi=150)
plt.close()
print(f"Saved: {PLOT_DIR}/12_bias_evolution_by_lr.png")


print("\n" + "=" * 70)
print("PERFORMANCE TABLES")
print("=" * 70)

# Training Summary Table (for the primary lr=0.01 run) ---
training_summary = pd.DataFrame({
    "Field": [
        "Dataset Size", "Train/Test Split", "Learning Rate", "Epochs Run",
        "Final Weights", "Final Bias", "Accuracy", "Precision", "Recall", "F1-score"
    ],
    "Value": [
        data.shape[0],
        f"{X_train.shape[0]}/{X_test.shape[0]} (80/20)",
        model.lr,
        len(model.history_errors),
        np.round(model.weights, 4).tolist(),
        round(model.bias, 4),
        round(acc, 4),
        round(prec, 4),
        round(rec, 4),
        round(f1, 4),
    ],
})
print("\nTraining Summary:")
print(training_summary.to_string(index=False))
training_summary.to_csv(f"{TABLE_DIR}/training_summary.csv", index=False)
print(f"Saved: {TABLE_DIR}/training_summary.csv")

# Epoch-wise Learning Table (for the primary lr=0.01 run) ---
epoch_table = pd.DataFrame({
    "Epoch": epochs_range,
    "Errors": model.history_errors,
    "Weight 1 (Variance)": [w[0] for w in model.history_weights],
    "Weight 2 (Skewness)": [w[1] for w in model.history_weights],
    "Bias": model.history_bias,
})
print("\nEpoch-wise Learning Table (first 10 rows):")
print(epoch_table.head(10).to_string(index=False))
epoch_table.to_csv(f"{TABLE_DIR}/epoch_wise_learning.csv", index=False)
print(f"Saved: {TABLE_DIR}/epoch_wise_learning.csv")


print("\n" + "=" * 70)
print("TASK 8: COMPARISON WITH SCIKIT-LEARN'S PERCEPTRON")
print("=" * 70)

sk_model = SklearnPerceptron(eta0=0.01, max_iter=40, random_state=SEED)
sk_model.fit(X_train, y_train)
sk_pred = sk_model.predict(X_test)

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "From-Scratch Perceptron": [
        round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)
    ],
    "Scikit-learn Perceptron": [
        round(accuracy_score(y_test, sk_pred), 4),
        round(precision_score(y_test, sk_pred), 4),
        round(recall_score(y_test, sk_pred), 4),
        round(f1_score(y_test, sk_pred), 4),
    ],
})
print(comparison.to_string(index=False))









