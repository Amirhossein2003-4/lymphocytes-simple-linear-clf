# Lymphocyte Detection in Histopathology Images (PathMNIST)

A binary classification pipeline built with Scikit-Learn to detect lymphocytes in colorectal histopathology images using the PathMNIST dataset (part of the MedMNIST v2 benchmark).

This project is a hands-on exercise inspired by *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (Aurélien Géron) — especially the chapters on classification, threshold tuning, and precision/recall trade-offs.

## Biological Background

PathMNIST is derived from real H&E-stained (Hematoxylin & Eosin) histological slides of colorectal cancer tissue. Each 28x28 RGB patch belongs to one of 9 tissue types, including tumor epithelium, stroma, muscle, debris, and immune cells.

Lymphocytes are a type of white blood cell and a key part of the adaptive immune system. In tumor histopathology, they often appear as tumor-infiltrating lymphocytes (TILs), which are actively studied in oncology because their presence can be associated with immune response and prognosis.

Detecting lymphocytes automatically is a clinically motivated computer vision problem, but also a naturally imbalanced and visually challenging classification task.

## Dataset

- **Source:** PathMNIST (MedMNIST v2)
- **Train samples:** 89,996
- **Feature representation:** flattened from `(89996, 28, 28, 3)` to `(89996, 2352)`
- **Target class:** `3 -> Lymphocytes`

### Class distribution (train)

| Class | Tissue | Count |
|------:|--------|------:|
| 0 | Adipose | 9,366 |
| 1 | Background | 9,509 |
| 2 | Debris | 10,360 |
| 3 | Lymphocytes | 10,401 |
| 4 | Mucus | 8,006 |
| 5 | Smooth muscle | 12,182 |
| 6 | Normal colon mucosa | 7,886 |
| 7 | Cancer-associated stroma | 9,401 |
| 8 | Colorectal adenocarcinoma epithelium | 12,885 |

Since the task is reframed as **Lymphocyte vs. Everything Else**, the positive class becomes a minority class at about **11.6%** of the data. This is the main reason why accuracy alone is not a reliable metric here.

## Pipeline

### Model 1: Linear SVM Baseline
- `StandardScaler -> SGDClassifier`
- `loss="hinge"` to approximate a linear SVM
- `class_weight="balanced"` to compensate for class imbalance
- 3-fold cross-validation for evaluation
- Out-of-fold decision scores used to tune the classification threshold without test-set leakage

### Model 2: Random Forest Baseline
- `RandomForestClassifier`
- `class_weight="balanced"`
- 3-fold cross-validation
- Out-of-fold probabilities used to tune the classification threshold without test-set leakage

## Results

## SGDClassifier / Linear SVM

### Default threshold (`0.0`)
This threshold corresponds to the raw decision function boundary.

| Metric | Negative (Non-Lymph) | Positive (Lymph) |
|--------|-----------------------|------------------|
| Precision | 0.9461 | 0.1388 |
| Recall | 0.6170 | 0.6372 |
| F1-score | 0.7469 | 0.2279 |

- **Accuracy:** 61.88%

### Cross-validation on train set
- **Mean Accuracy:** 70.48%
- **Mean F1 (positive class):** 36.74%

### Optimal threshold
The threshold was tuned using out-of-fold predictions to maximize F1.

- **Optimal threshold:** `1.4971`
- **F1 at optimal threshold (CV/train):** 0.3693

| Metric | Negative (Non-Lymph) | Positive (Lymph) |
|--------|-----------------------|------------------|
| Precision | 0.9440 | 0.1382 |
| Recall | 0.6283 | 0.6151 |
| F1-score | 0.7545 | 0.2256 |

- **Accuracy:** 62.72%

---

## Random Forest

### Default threshold (`0.50`)
| Metric | Negative (Non-Lymph) | Positive (Lymph) |
|--------|-----------------------|------------------|
| Precision | 0.9143 | 0.3472 |
| Recall | 0.9928 | 0.0394 |
| F1-score | 0.9520 | 0.0708 |

- **Accuracy:** 90.86%

### Cross-validation on train set
- **Mean Accuracy:** 92.27%
- **Mean F1 (positive class):** 51.45%
- **Mean ROC AUC:** 94.79%
- **Mean Average Precision:** 75.96%

### Out-of-fold train evaluation
- **ROC AUC:** 0.9479
- **Average Precision:** 0.7595

### Optimal threshold
The threshold was tuned using out-of-fold probabilities to maximize F1.

- **Optimal threshold:** `0.2600`
- **F1 at optimal threshold (CV/train):** 0.6628

| Metric | Negative (Non-Lymph) | Positive (Lymph) |
|--------|-----------------------|------------------|
| Precision | 0.9280 | 0.2280 |
| Recall | 0.9115 | 0.2697 |
| F1-score | 0.9197 | 0.2471 |

- **Accuracy:** 85.49%

### Test-set ranking metrics
- **ROC AUC:** 0.6782
- **Average Precision:** 0.1834

## Comparative Summary

| Model | CV Accuracy | CV F1 (positive) | OOF Train ROC AUC | Test F1 (optimal) | Test ROC AUC | Notes |
|------|-------------:|-----------------:|------------------:|------------------:|-------------:|------|
| SGD / Linear SVM | 70.48% | 36.74% | 0.7573 | 0.2256 | 0.6496 | Stable baseline, but limited by linear decision boundary |
| Random Forest | 92.27% | 51.45% | 0.9479 | 0.2471 | 0.6782 | Strong CV performance, but test generalization is much weaker |

## Why Accuracy Is Misleading Here

At first glance, accuracy can look decent — especially for Random Forest — but this is not the right metric for this problem.

Because lymphocytes are the minority class, a naive classifier that predicts "not lymphocyte" for everything can already achieve high accuracy simply by exploiting class imbalance. In this setting, the more meaningful metrics are:

- **Recall** for lymphocytes: how many actual lymphocytes are detected
- **Precision** for lymphocytes: how many predicted lymphocytes are correct
- **F1-score**: the balance between precision and recall
- **ROC AUC / Average Precision**: ranking quality, especially under imbalance

For the SGD baseline, class weighting helped improve minority-class recall, but precision remained low. Threshold tuning changed the operating point slightly, but did not fundamentally solve the problem.

For Random Forest, cross-validation scores were very strong, but the test results dropped substantially, which suggests **overfitting** and weak generalization.

## Limitations

This project is a useful baseline, but it has several important limitations:

1. **Flattened pixels lose spatial structure**  
   The images were converted into 1D feature vectors, so classical models do not learn local texture patterns, shapes, or tissue context the way CNNs do.

2. **Very small image size**  
   Each sample is only `28x28`, which is extremely low resolution for histopathology. Fine-grained visual details may be lost.

3. **Visual similarity between classes**  
   Some tissue classes in PathMNIST can be highly similar at this resolution. For example:
   - lymphocytes vs. background
   - lymphocytes vs. debris
   - lymphocytes vs. small dark nuclei in stroma or epithelium
   - lymphocytes vs. cancer-associated stroma with dense cellular texture  
   This makes the one-vs-rest task harder than a simple binary label suggests.

4. **Class imbalance**  
   The positive class is only about 11.6% of the dataset, so the model can be biased toward the majority class even with `class_weight="balanced"`.

5. **Potential label ambiguity / patch-level uncertainty**  
   In histopathology, some patches may contain mixed tissue appearance or ambiguous boundaries, which can introduce noise into the labels.

6. **Random Forest overfitting**  
   The large gap between cross-validation performance and test performance suggests that the model is fitting patterns in the training folds that do not transfer well to unseen data.

7. **Threshold tuning cannot fix representation limits**  
   Optimizing the classification threshold helps choose a better operating point, but it cannot compensate for a weak feature representation or a fundamentally limited model family.

## Interpretation

The key takeaway is that this is a **challenging histopathology classification task**, not just a simple binary classification problem.

- The linear SVM baseline is easy to train and gives a reasonable starting point.
- Random Forest performs well in cross-validation but does not generalize as strongly to the test set.
- The main bottleneck is likely the combination of:
  - low image resolution
  - flattened input representation
  - strong class imbalance
  - visual similarity between multiple tissue classes

In practice, a more suitable next step would be a CNN-based approach that preserves spatial structure and can learn tissue-specific features directly from the images.

## Next Steps

- Extend this pipeline to a full multi-class classifier over all 9 tissue types.
- Replace flattened features with a CNN or transfer-learning model.
- Add data augmentation to improve robustness.
- Investigate class-specific confusion patterns to see which tissues are most frequently mistaken for lymphocytes.
- Explore resampling methods such as SMOTE or undersampling for imbalance handling.
- Compare this classical ML baseline against a deep learning pipeline.

## Author

**Amirhossein**  
Email: amirhossein070905@gmail.com  
Telegram: @itsamirhosseingadimi
