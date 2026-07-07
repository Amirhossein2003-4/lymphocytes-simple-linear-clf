"""
Binary Classification of Lymphocytes in Histopathology Images (PathMNIST)
===========================================================================

This script trains a linear SVM (via SGDClassifier) to detect a single
tissue class -- Lymphocytes -- against all other tissue types in the
PathMNIST dataset (a colorectal histopathology image collection from
the MedMNIST benchmark).

Key techniques demonstrated:
    - Reformulating a multi-class problem as a one-vs-rest binary task.
    - Feature scaling as a prerequisite for gradient-based linear models.
    - Handling class imbalance via `class_weight="balanced"`.
    - Threshold tuning: choosing a decision threshold that maximizes F1
      instead of relying on the naive default of 0.
    - Cross-validated evaluation to avoid overly optimistic (or misleading)
      accuracy-only reporting.
"""

import medmnist
from medmnist import INFO

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    classification_report,
)

# ============================================================
# Configuration
# ============================================================

DATA_FLAG = "pathmnist"
POSITIVE_CLASS = 3     # Target class: Lymphocytes
CV_FOLDS = 3
N_JOBS = -1            # Use all available CPU cores
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.0

CLASS_NAMES = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth muscle",
    6: "normal colon mucosa",
    7: "cancer-associated stroma",
    8: "colorectal adenocarcinoma epithelium",
}

# ============================================================
# Dataset Loading
# ============================================================

info = INFO[DATA_FLAG]
DataClass = getattr(medmnist, info["python_class"])

train_dataset = DataClass(split="train", download=True)
val_dataset = DataClass(split="val", download=True)
test_dataset = DataClass(split="test", download=True)

# ============================================================
# Helper Functions
# ============================================================


def prepare_dataset(dataset):
    """
    Convert a MedMNIST dataset split into flat NumPy arrays.

    Each (28, 28, 3) image is flattened into a 1D vector of length 2352,
    since SGDClassifier expects tabular (n_samples, n_features) input
    rather than image tensors.
    """
    X, y = [], []
    for img, label in dataset:
        X.append(np.asarray(img, dtype=np.float32).reshape(-1))
        y.append(int(label[0]))
    return np.asarray(X), np.asarray(y)


def print_dataset_info(X, y, split_name):
    """Print shape and class distribution for a quick sanity check."""
    print(f"\n{'=' * 60}")
    print(f"{split_name.upper()} DATASET INFO")
    print(f"{'=' * 60}")
    print(f"X shape: {X.shape} | y shape: {y.shape}")
    print("Class distribution:")
    print(Counter(y))


def find_optimal_threshold_f1(y_true, scores):
    """
    Find the decision-function threshold that maximizes F1-score.

    IMPORTANT: `scores` must come from cross-validated (out-of-fold)
    predictions on the TRAINING set. Using the test set here would leak
    test information into model selection and invalidate the final
    evaluation.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)

    # precision_recall_curve returns one extra point (for threshold = +inf)
    # compared to `thresholds`, so we drop the last element to align arrays.
    precisions, recalls = precisions[:-1], recalls[:-1]

    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0,
    )

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx], best_idx


def evaluate_at_threshold(y_true, scores, threshold, label):
    """
    Apply a custom decision threshold to raw decision-function scores
    and print accuracy, classification report, and confusion matrix.

    Centralizing this logic avoids duplicating (and potentially
    mis-indexing) evaluation code for the default vs. optimal thresholds.
    """
    y_pred = scores > threshold

    print(f"\n{'=' * 60}")
    print(f"EVALUATION — {label} (threshold = {threshold:.4f})")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.6f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return y_pred


# --- Plotting Helpers ---


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Precision and Recall vs Decision Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(precisions, recalls, thresholds=None):
    plt.figure(figsize=(7, 6))
    plt.plot(recalls, precisions, "b-", linewidth=2, label="Precision-Recall Curve")

    if thresholds is not None:
        step = max(1, len(thresholds) // 15)
        for i in range(0, len(thresholds), step):
            plt.annotate(
                f"{thresholds[i]:.1f}",
                (recalls[i], precisions[i]),
                fontsize=8,
                alpha=0.7,
                xytext=(5, 5),
                textcoords="offset points",
            )

    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_roc_curve(fpr, tpr, auc_score):
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, "r-", linewidth=2, label=f"SGDClassifier (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR / Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_threshold_distribution(scores, default_threshold, optimal_threshold):
    plt.figure(figsize=(10, 4))
    plt.hist(scores, bins=100, color="gray", alpha=0.5, label="Score Distribution")
    plt.axvline(
        x=default_threshold, color="red", linestyle="--",
        label=f"Default Threshold ({default_threshold})",
    )
    plt.axvline(
        x=optimal_threshold, color="green", linestyle="-", linewidth=2,
        label=f"Optimal Threshold ({optimal_threshold:.2f})",
    )
    plt.title("Distribution of Decision Scores on Test Set")
    plt.xlabel("Decision Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# Build Datasets
# ============================================================

X_train, y_train = prepare_dataset(train_dataset)
X_val, y_val = prepare_dataset(val_dataset)
X_test, y_test = prepare_dataset(test_dataset)

print_dataset_info(X_train, y_train, "train")
print(f"\nTarget Class: {POSITIVE_CLASS} -> {CLASS_NAMES[POSITIVE_CLASS]}")

# ============================================================
# Binary Targets: Lymphocyte vs. Non-Lymphocyte
# ============================================================

y_train_binary = y_train == POSITIVE_CLASS
y_val_binary = y_val == POSITIVE_CLASS
y_test_binary = y_test == POSITIVE_CLASS

# ============================================================
# Pipeline Definition (Scaling + Classification)
# ============================================================

# Why scaling matters:
# SGD is highly sensitive to feature scale. Raw pixel values in [0, 255]
# distort the loss surface and lead to slow/incomplete convergence and
# a suboptimal decision boundary. StandardScaler (mean=0, var=1) fixes
# this. `class_weight="balanced"` compensates for the fact that
# lymphocytes are a small minority among the 9 tissue classes.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd_clf", SGDClassifier(
        loss="hinge",              # Hinge loss -> linear SVM
        penalty="l2",
        alpha=0.0001,
        max_iter=1000,
        tol=1e-3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
    )),
])

# ============================================================
# Train the Pipeline
# ============================================================

print("\nTraining the Pipeline (StandardScaler + SGDClassifier)...")
pipeline.fit(X_train, y_train_binary)

# ============================================================
# Evaluation on Test Set — Default Threshold (0.0)
# ============================================================

test_decision_scores = pipeline.decision_function(X_test)

evaluate_at_threshold(
    y_test_binary, test_decision_scores, DEFAULT_THRESHOLD, label="DEFAULT MODEL"
)

# ============================================================
# Cross-Validated Evaluation
# ============================================================

print(f"\nRunning {CV_FOLDS}-Fold Cross-Validation (n_jobs={N_JOBS})...")

cv_accuracy = cross_val_score(
    pipeline, X_train, y_train_binary, cv=CV_FOLDS, scoring="accuracy", n_jobs=N_JOBS
)
cv_f1 = cross_val_score(
    pipeline, X_train, y_train_binary, cv=CV_FOLDS, scoring="f1", n_jobs=N_JOBS
)

print(f"Mean CV Accuracy: {cv_accuracy.mean():.6f} (Scores: {cv_accuracy})")
print(f"Mean CV F1 Score: {cv_f1.mean():.6f} (Scores: {cv_f1})")

# Note: on an imbalanced binary problem (lymphocytes are ~1/9 of samples),
# accuracy alone is misleading -- a model that always predicts "negative"
# can still score ~85-90% accuracy. F1 is the metric that actually reflects
# how well the positive (minority) class is detected.

# ============================================================
# Cross-Validated Decision Scores & Curves
# ============================================================

print("\nGenerating out-of-fold decision scores for metric curves...")

y_scores = cross_val_predict(
    pipeline, X_train, y_train_binary, cv=CV_FOLDS,
    method="decision_function", n_jobs=N_JOBS,
)

precisions, recalls, pr_thresholds = precision_recall_curve(y_train_binary, y_scores)
fpr, tpr, roc_thresholds = roc_curve(y_train_binary, y_scores)
auc_score = roc_auc_score(y_train_binary, y_scores)

print("\nPlotting performance curves...")
plot_precision_recall_vs_threshold(precisions, recalls, pr_thresholds)
plot_precision_recall_curve(precisions, recalls, pr_thresholds)
plot_roc_curve(fpr, tpr, auc_score)

# ============================================================
# Find & Evaluate the Optimal Threshold (F1-Maximizing)
# ============================================================

# Instead of guessing a threshold manually, we compute the one that
# maximizes F1 on cross-validated (out-of-fold) train scores, then apply
# it once to the held-out test set for a fair, leak-free evaluation.
optimal_threshold, optimal_f1_cv, _ = find_optimal_threshold_f1(y_train_binary, y_scores)

print(f"\n{'=' * 60}")
print("OPTIMAL THRESHOLD SEARCH (based on CV out-of-fold scores)")
print(f"{'=' * 60}")
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"F1 at this threshold (CV/train): {optimal_f1_cv:.4f}")

y_test_optimal_pred = evaluate_at_threshold(
    y_test_binary, test_decision_scores, optimal_threshold, label="OPTIMAL MODEL"
)

# ============================================================
# Visualize the Decision Boundary Shift (Default vs. Optimal)
# ============================================================

plot_threshold_distribution(test_decision_scores, DEFAULT_THRESHOLD, optimal_threshold)

print("\nProcess finished successfully.")


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
)

# ============================================================
# Random Forest Baseline
# ============================================================

print(f"\n{'=' * 60}")
print("RANDOM FOREST MODEL")
print(f"{'=' * 60}")

forest_clf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced",
)

# ------------------------------------------------------------
# Train on full training set
# ------------------------------------------------------------

print("\nTraining RandomForestClassifier...")
forest_clf.fit(X_train, y_train_binary)

# ------------------------------------------------------------
# Test-set probabilities
# ------------------------------------------------------------

# RandomForest has no decision_function(), so we use predict_proba().
# Column 1 corresponds to the positive class: lymphocytes=True
test_probas_forest = forest_clf.predict_proba(X_test)[:, 1]

# Default probability threshold = 0.5
DEFAULT_PROBA_THRESHOLD = 0.5

y_test_pred_forest_default = test_probas_forest >= DEFAULT_PROBA_THRESHOLD

print(f"\n{'=' * 60}")
print(f"RANDOM FOREST — TEST EVALUATION (threshold = {DEFAULT_PROBA_THRESHOLD:.2f})")
print(f"{'=' * 60}")
print(f"Accuracy: {accuracy_score(y_test_binary, y_test_pred_forest_default):.6f}")
print("\nClassification Report:")
print(classification_report(y_test_binary, y_test_pred_forest_default, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test_binary, y_test_pred_forest_default))

# ------------------------------------------------------------
# Cross-validated metrics on training set
# ------------------------------------------------------------

print(f"\nRunning {CV_FOLDS}-Fold Cross-Validation for Random Forest (n_jobs={N_JOBS})...")

cv_accuracy_forest = cross_val_score(
    forest_clf,
    X_train,
    y_train_binary,
    cv=CV_FOLDS,
    scoring="accuracy",
    n_jobs=N_JOBS,
)

cv_f1_forest = cross_val_score(
    forest_clf,
    X_train,
    y_train_binary,
    cv=CV_FOLDS,
    scoring="f1",
    n_jobs=N_JOBS,
)

cv_roc_auc_forest = cross_val_score(
    forest_clf,
    X_train,
    y_train_binary,
    cv=CV_FOLDS,
    scoring="roc_auc",
    n_jobs=N_JOBS,
)

cv_avg_precision_forest = cross_val_score(
    forest_clf,
    X_train,
    y_train_binary,
    cv=CV_FOLDS,
    scoring="average_precision",
    n_jobs=N_JOBS,
)

print(f"Mean CV Accuracy:          {cv_accuracy_forest.mean():.6f} (Scores: {cv_accuracy_forest})")
print(f"Mean CV F1 Score:          {cv_f1_forest.mean():.6f} (Scores: {cv_f1_forest})")
print(f"Mean CV ROC AUC:           {cv_roc_auc_forest.mean():.6f} (Scores: {cv_roc_auc_forest})")
print(f"Mean CV Average Precision: {cv_avg_precision_forest.mean():.6f} (Scores: {cv_avg_precision_forest})")

# ------------------------------------------------------------
# Cross-validated out-of-fold probabilities
# ------------------------------------------------------------

print("\nGenerating out-of-fold probabilities for Random Forest...")

y_probas_forest = cross_val_predict(
    forest_clf,
    X_train,
    y_train_binary,
    cv=CV_FOLDS,
    method="predict_proba",
    n_jobs=N_JOBS,
)

# Positive-class probabilities
y_scores_forest = y_probas_forest[:, 1]

# ------------------------------------------------------------
# PR / ROC curves
# ------------------------------------------------------------

precisions_forest, recalls_forest, pr_thresholds_forest = precision_recall_curve(
    y_train_binary, y_scores_forest
)

fpr_forest, tpr_forest, roc_thresholds_forest = roc_curve(
    y_train_binary, y_scores_forest
)

auc_score_forest = roc_auc_score(y_train_binary, y_scores_forest)
avg_precision_forest = average_precision_score(y_train_binary, y_scores_forest)

print(f"\nRandom Forest ROC AUC (OOF train): {auc_score_forest:.6f}")
print(f"Random Forest Average Precision (OOF train): {avg_precision_forest:.6f}")

print("\nPlotting Random Forest performance curves...")
plot_precision_recall_vs_threshold(precisions_forest, recalls_forest, pr_thresholds_forest)
plot_precision_recall_curve(precisions_forest, recalls_forest, pr_thresholds_forest)
plot_roc_curve(fpr_forest, tpr_forest, auc_score_forest)

# ------------------------------------------------------------
# Find optimal threshold for F1
# ------------------------------------------------------------

optimal_threshold_forest, optimal_f1_cv_forest, _ = find_optimal_threshold_f1(
    y_train_binary,
    y_scores_forest,
)

print(f"\n{'=' * 60}")
print("RANDOM FOREST OPTIMAL THRESHOLD SEARCH (based on CV OOF probs)")
print(f"{'=' * 60}")
print(f"Optimal Probability Threshold: {optimal_threshold_forest:.4f}")
print(f"F1 at this threshold (CV/train): {optimal_f1_cv_forest:.4f}")

# ------------------------------------------------------------
# Evaluate Random Forest on test set using optimal threshold
# ------------------------------------------------------------

y_test_pred_forest_optimal = test_probas_forest >= optimal_threshold_forest

print(f"\n{'=' * 60}")
print(f"RANDOM FOREST — TEST EVALUATION (optimal threshold = {optimal_threshold_forest:.4f})")
print(f"{'=' * 60}")
print(f"Accuracy: {accuracy_score(y_test_binary, y_test_pred_forest_optimal):.6f}")
print("\nClassification Report:")
print(classification_report(y_test_binary, y_test_pred_forest_optimal, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test_binary, y_test_pred_forest_optimal))

# ------------------------------------------------------------
# Test-set ROC AUC / AP
# ------------------------------------------------------------

test_auc_forest = roc_auc_score(y_test_binary, test_probas_forest)
test_avg_precision_forest = average_precision_score(y_test_binary, test_probas_forest)

print(f"\nRandom Forest Test ROC AUC: {test_auc_forest:.6f}")
print(f"Random Forest Test Average Precision: {test_avg_precision_forest:.6f}")

# Optional: visualize probability distribution
plt.figure(figsize=(10, 4))
plt.hist(test_probas_forest, bins=50, color="steelblue", alpha=0.7, label="RF probability distribution")
plt.axvline(
    x=DEFAULT_PROBA_THRESHOLD,
    color="red",
    linestyle="--",
    label=f"Default Threshold ({DEFAULT_PROBA_THRESHOLD})",
)
plt.axvline(
    x=optimal_threshold_forest,
    color="green",
    linestyle="-",
    linewidth=2,
    label=f"Optimal Threshold ({optimal_threshold_forest:.2f})",
)
plt.title("Distribution of Random Forest Positive-Class Probabilities on Test Set")
plt.xlabel("Predicted Probability of Lymphocytes")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# Comparative Summary: SGD vs Random Forest
# ============================================================

print(f"\n{'=' * 60}")
print("COMPARATIVE SUMMARY — SGD (Linear SVM) vs Random Forest")
print(f"{'=' * 60}")

# --- SGD test metrics at default threshold ---
sgd_test_pred_default = test_decision_scores > DEFAULT_THRESHOLD
sgd_test_acc_default = accuracy_score(y_test_binary, sgd_test_pred_default)
sgd_test_auc = roc_auc_score(y_test_binary, test_decision_scores)
sgd_test_ap = average_precision_score(y_test_binary, test_decision_scores)

# --- SGD test metrics at optimal threshold ---
sgd_test_pred_optimal = test_decision_scores > optimal_threshold
sgd_test_f1_default = f1_score(y_test_binary, sgd_test_pred_default)
sgd_test_f1_optimal = f1_score(y_test_binary, sgd_test_pred_optimal)

# --- RF test metrics ---
rf_test_acc_default = accuracy_score(y_test_binary, y_test_pred_forest_default)
rf_test_acc_optimal = accuracy_score(y_test_binary, y_test_pred_forest_optimal)
rf_test_f1_default = f1_score(y_test_binary, y_test_pred_forest_default)
rf_test_f1_optimal = f1_score(y_test_binary, y_test_pred_forest_optimal)

print("\nSGD / Linear SVM:")
print(f"  CV Accuracy Mean:          {cv_accuracy.mean():.6f}")
print(f"  CV F1 Mean:                {cv_f1.mean():.6f}")
print(f"  OOF Train ROC AUC:         {auc_score:.6f}")
print(f"  Test Accuracy @ default:   {sgd_test_acc_default:.6f}")
print(f"  Test F1 @ default:         {sgd_test_f1_default:.6f}")
print(f"  Test F1 @ optimal:         {sgd_test_f1_optimal:.6f}")
print(f"  Test ROC AUC:              {sgd_test_auc:.6f}")
print(f"  Test Average Precision:    {sgd_test_ap:.6f}")
print(f"  Optimal threshold:         {optimal_threshold:.6f}")

print("\nRandom Forest:")
print(f"  CV Accuracy Mean:          {cv_accuracy_forest.mean():.6f}")
print(f"  CV F1 Mean:                {cv_f1_forest.mean():.6f}")
print(f"  OOF Train ROC AUC:         {auc_score_forest:.6f}")
print(f"  Test Accuracy @ default:   {rf_test_acc_default:.6f}")
print(f"  Test Accuracy @ optimal:   {rf_test_acc_optimal:.6f}")
print(f"  Test F1 @ default:         {rf_test_f1_default:.6f}")
print(f"  Test F1 @ optimal:         {rf_test_f1_optimal:.6f}")
print(f"  Test ROC AUC:              {test_auc_forest:.6f}")
print(f"  Test Average Precision:    {test_avg_precision_forest:.6f}")
print(f"  Optimal threshold:         {optimal_threshold_forest:.6f}")




















