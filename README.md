# Lymphocyte Detection in Histopathology Images (PathMNIST)

A binary classification pipeline built with **Scikit-Learn** to detect
**Lymphocytes** in colorectal histopathology images, using the
[PathMNIST](https://medmnist.com/) dataset (part of the MedMNIST v2 benchmark).

This project is a hands-on exercise following **"Hands-On Machine Learning
with Scikit-Learn, Keras & TensorFlow"** (Aurélien Géron) — specifically the
chapter on classification, threshold tuning, and precision/recall trade-offs.

---

## Biological Background

PathMNIST is derived from real H&E-stained (Hematoxylin & Eosin) histological
slides of **colorectal cancer tissue**. Each 28x28 RGB patch belongs to one
of 9 tissue types, including tumor epithelium, stroma, muscle, and immune
cells.

**Lymphocytes** are a type of white blood cell (part of the adaptive immune
system) that infiltrate tumor tissue as part of the body's immune response
to cancer. The density and distribution of tumor-infiltrating lymphocytes
(TILs) is an active area of research in oncology, since higher TIL counts
are often associated with better prognosis in colorectal cancer. Automating
their detection at scale is a real, clinically-motivated computer vision
problem — and also a naturally **imbalanced classification task**, since
lymphocytes make up a small fraction of tissue compared to epithelium,
stroma, and muscle.

---

## Dataset

- **Source:** PathMNIST (MedMNIST v2)
- **Train samples:** 89,996 — flattened to `(89996, 2352)` feature vectors
  (28×28×3 pixels)
- **Target class:** `3` → Lymphocytes
- **Class distribution (train):**

  | Class | Tissue | Count |
  |---|---|---|
  | 0 | Adipose | 9,366 |
  | 1 | Background | 9,509 |
  | 2 | Debris | 10,360 |
  | **3** | **Lymphocytes** | **10,401** |
  | 4 | Mucus | 8,006 |
  | 5 | Smooth muscle | 12,182 |
  | 6 | Normal colon mucosa | 7,886 |
  | 7 | Cancer-associated stroma | 9,401 |
  | 8 | Colorectal adenocarcinoma epithelium | 12,885 |

Since the task is reframed as **Lymphocyte vs. Everything Else**, the
positive class becomes a true minority (~11.5% of samples), which is what
drives most of the results below.

---

## Pipeline

- `StandardScaler` → `SGDClassifier` (hinge loss = linear SVM)
- `class_weight="balanced"` to compensate for class imbalance
- 3-Fold cross-validation for model evaluation
- Out-of-fold decision scores used to tune the classification threshold
  (avoids data leakage into the test set)

---

## Results

### Default Threshold (0.0)

| Metric | Negative (Non-Lymph) | Positive (Lymph) |
|---|---|---|
| Precision | 0.9461 | 0.1388 |
| Recall | 0.6170 | 0.6372 |
| F1-score | 0.7469 | 0.2279 |

**Accuracy: 61.88%**

### Cross-Validation (Train Set)

- Mean Accuracy: **70.48%**
- Mean F1 (positive class): **36.74%**

### Optimal Threshold (F1-maximizing, found via CV, threshold ≈ 1.497)

| Metric | Negative (Non-Lymph) | Positive (Lymph) |
|---|---|---|
| Precision | 0.9440 | 0.1382 |
| Recall | 0.6283 | 0.6151 |
| F1-score | 0.7545 | 0.2256 |

**Accuracy: 62.72%**

### Why Accuracy Is Misleading Here

At first glance, ~62-70% accuracy seems mediocre — but the real story is in
the **per-class breakdown**. Because lymphocytes are the minority class
(~11.5% of samples), a naive classifier that predicts "not lymphocyte" for
everything would already score ~88% accuracy while detecting **zero**
lymphocytes.

Here, `class_weight="balanced"` intentionally trades overall accuracy for
better recall on the minority class: the model correctly identifies **~62%
of actual lymphocytes** (recall), at the cost of a high false-positive rate
(precision ~14%). This is reflected in the low F1-score (~0.23) for the
positive class, which is the metric that actually matters for this
imbalanced problem — not the headline accuracy number.

Threshold tuning (moving from 0.0 to 1.497) gave a marginal shift toward
precision over recall, but did not meaningfully change the F1-score
(0.2279 → 0.2256), suggesting the linear decision boundary itself — not the
threshold — is the current bottleneck.

---

## Next Steps

- Extend this pipeline to a **full multi-class classifier** across all 9
  tissue types and benchmark it against this one-vs-rest binary approach.
- Explore non-linear models (e.g. `SGDClassifier` with kernel approximation,
  or tree-based ensembles) since a linear boundary appears to cap
  performance on raw pixel features.

---

## Author

**Amirhossein**
📧 amirhossein070905@gmail.com
💬 Telegram: [@itsamirhosseingadimi](https://t.me/itsamirhosseingadimi)
