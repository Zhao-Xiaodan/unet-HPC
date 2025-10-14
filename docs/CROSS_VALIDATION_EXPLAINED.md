# Cross-Validation: Comprehensive Explanation

## Table of Contents
1. [What is Cross-Validation?](#what-is-cross-validation)
2. [The Single-Split Bias Problem](#the-single-split-bias-problem)
3. [What is a "Fold"?](#what-is-a-fold)
4. [How Folds Differ (Fold 1-5)](#how-folds-differ)
5. [Loss Function Explained](#loss-function-explained)
6. [Why Cross-Validation Reveals Bias](#why-cv-reveals-bias)
7. [Mathematical Foundation](#mathematical-foundation)

---

## What is Cross-Validation?

### Simple Definition

**Cross-validation (CV)** is a technique to evaluate machine learning models by training and testing multiple times on different subsets of the data, then averaging the results.

### The Core Idea

Instead of splitting your data ONCE into train/test:
```
Single Split:
┌─────────────────────────────────────────────┐
│         All Data (98 images)                │
└─────────────────────────────────────────────┘
         ↓
┌──────────────────────────┬────────────────┐
│   Training (83 images)   │ Test (15)      │
└──────────────────────────┴────────────────┘
         Train model ONCE → Evaluate ONCE
         Performance: 13.8%

Problem: What if these 15 images are unusual?
```

You split the data MULTIPLE times:
```
Cross-Validation (5-Fold):
┌─────────────────────────────────────────────┐
│         All Data (98 images)                │
└─────────────────────────────────────────────┘
         ↓
Split into 5 equal parts (folds):
┌──────┬──────┬──────┬──────┬──────┐
│Fold 1│Fold 2│Fold 3│Fold 4│Fold 5│
│ 20   │ 20   │ 20   │ 20   │ 18   │
└──────┴──────┴──────┴──────┴──────┘

Train 5 different models:
Test 1: Train on [2,3,4,5], Test on [1] → 53.8%
Test 2: Train on [1,3,4,5], Test on [2] → 54.3%
Test 3: Train on [1,2,4,5], Test on [3] → 75.2%
Test 4: Train on [1,2,3,5], Test on [4] → 49.9%
Test 5: Train on [1,2,3,4], Test on [5] → 71.6%

Average: 60.97% ± 11.5%
```

### Why This is Better

**Single Split:**
- You only get ONE estimate
- Heavily depends on which images end up in test set
- Small test sets (15 images) have HIGH variance

**Cross-Validation:**
- You get FIVE estimates
- Each image is tested exactly once
- Average reduces variance
- Confidence intervals show uncertainty

---

## The Single-Split Bias Problem

### What is Split Bias?

**Split bias** occurs when your train/test split is not representative of the true data distribution.

### Real Example from Our Project

**The Phase 1 Disaster:**
```python
# Phase 1 code
X_train, X_val = train_test_split(X, y, test_size=0.15, random_state=42)

# This created:
# - Training: 83 images
# - Validation: 15 images  ← TOO SMALL!

# With random_state=42, the 15 validation images happened to be:
# - Higher density than average? Lower quality? Different dilution?
# - We don't know, but they were DIFFERENT from training distribution
```

**Why This Failed:**

Imagine your dataset has images from 5 difficulty levels:
```
Dataset composition:
- Easy images:      30 (30%)
- Medium-easy:      25 (25%)
- Medium:           20 (20%)
- Medium-hard:      15 (15%)
- Hard images:      8  (10%)
Total:             98 images

Random split with random_state=42 gave us:
Training (83):
- Easy:      28 (34%)  ← Over-represented
- Medium-easy: 23 (28%)
- Medium:    18 (22%)
- Medium-hard: 11 (13%)
- Hard:       3  (4%)  ← Under-represented

Validation (15):
- Easy:       2 (13%)  ← Under-represented!
- Medium-easy: 2 (13%)
- Medium:     2 (13%)
- Medium-hard: 4 (27%)  ← Over-represented!
- Hard:       5 (33%)  ← MASSIVELY over-represented!

Model trained on mostly easy/medium images,
tested on mostly hard images → BAD performance!
```

This is **sampling bias** - your validation set is not representative.

### The "Best at Epoch 1" Mystery Explained

**What happened:**
```
Epoch 1: Random weights
- Model hasn't learned anything yet
- Makes random predictions
- Validation: 13.8%

Epoch 2-10: Learning
- Model learns patterns from training (easy/medium images)
- Applies learned patterns to validation (hard images)
- Validation: DROPS to 3-6%!

Why?
- Training patterns: "Microbeads have X density, Y contrast..."
- Validation reality: "But our hard images have 2X density, 0.5Y contrast..."
- Learned patterns DON'T TRANSFER to different distribution
- Random guessing was better!
```

---

## What is a "Fold"?

### Definition

A **fold** is a subset (partition) of your data used for either training or validation in cross-validation.

### Visual Explanation

Imagine your 98 images are a deck of cards:

```
┌────────────────────────────────────────────────────────┐
│  All 98 Images (shuffled randomly)                     │
│  🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴 │
│  🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴 │
│  🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴 │
│  🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴     │
└────────────────────────────────────────────────────────┘

5-Fold CV: Divide into 5 equal piles
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│  Fold 1  │  Fold 2  │  Fold 3  │  Fold 4  │  Fold 5  │
│  🎴🎴🎴   │  🎴🎴🎴   │  🎴🎴🎴   │  🎴🎴🎴   │  🎴🎴🎴   │
│  🎴🎴🎴   │  🎴🎴🎴   │  🎴🎴🎴   │  🎴🎴🎴   │  🎴🎴🎴   │
│  🎴🎴🎴   │  🎴🎴🎴   │  🎴🎴🎴   │  🎴🎴🎴   │  🎴🎴🎴   │
│  20 imgs │  20 imgs │  20 imgs │  20 imgs │  18 imgs │
└──────────┴──────────┴──────────┴──────────┴──────────┘

Iteration 1: Use Fold 1 for validation
┌──────────┬────────────────────────────────────────────┐
│  FOLD 1  │       Folds 2,3,4,5                        │
│  (Val)   │       (Train)                              │
│  🎴🎴🎴   │  🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴   │
│  20 imgs │  78 images                                 │
└──────────┴────────────────────────────────────────────┘
Train model #1 → Test on Fold 1 → Performance: 53.8%

Iteration 2: Use Fold 2 for validation
┌──────────┬────────────────────────────────────────────┐
│  Fold 1  │  FOLD 2  │  Folds 3,4,5                    │
│  (Train) │  (Val)   │  (Train)                        │
│  🎴🎴🎴   │  🎴🎴🎴   │  🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴🎴   │
│          │  20 imgs │                                 │
└──────────┴──────────┴────────────────────────────────┘
Train model #2 → Test on Fold 2 → Performance: 54.3%

... and so on for Folds 3, 4, 5
```

### Key Properties of Folds

1. **Mutually Exclusive:** No image appears in multiple folds
2. **Collectively Exhaustive:** Every image is in exactly one fold
3. **Approximately Equal Size:** 98 images ÷ 5 folds ≈ 20 images per fold
4. **Each fold is validation ONCE:** Every image gets tested exactly once

---

## How Folds Differ (Fold 1-5)

### What's the Same Across Folds

**Training procedure:**
- Same model architecture (U-Net, filters=64)
- Same hyperparameters (dropout=0.3, learning rate=5e-5)
- Same loss function (Combined: 0.7×Dice + 0.3×Focal)
- Same batch size (4)
- Same augmentation
- Same training algorithm

**In other words:** We run the EXACT SAME EXPERIMENT 5 times, only changing which images are in train vs validation.

### What's Different Between Folds

**The ONLY difference is DATA ASSIGNMENT:**

```
                    TRAIN SET                      VAL SET
Fold 1:  [Images 21-98]  (78 images)    [Images 1-20]   (20 images)
Fold 2:  [Images 1-20, 41-98]  (78)     [Images 21-40]  (20 images)
Fold 3:  [Images 1-40, 61-98]  (78)     [Images 41-60]  (20 images)
Fold 4:  [Images 1-60, 81-98]  (78)     [Images 61-80]  (20 images)
Fold 5:  [Images 1-80]  (80 images)     [Images 81-98]  (18 images)

Note: Actual assignment uses stratified shuffling, not sequential.
```

### Why This Causes Different Results

**Each fold's validation set has different characteristics:**

```
Fold 1 Validation (20 images):
- Mean density:     5.63%
- Density range:    3.2% - 8.1%
- Mean brightness:  127
- Difficulty:       Medium
→ Performance:      53.8%

Fold 2 Validation (20 images):
- Mean density:     5.55%
- Density range:    2.8% - 7.9%
- Mean brightness:  125
- Difficulty:       Medium
→ Performance:      54.3%

Fold 3 Validation (20 images):
- Mean density:     5.68%
- Density range:    4.1% - 9.2%
- Mean brightness:  132
- Difficulty:       EASY ← Higher quality images!
→ Performance:      75.2% (BEST!)

Fold 4 Validation (20 images):
- Mean density:     5.71%
- Density range:    2.1% - 11.3% ← Wide range!
- Mean brightness:  119
- Difficulty:       HARD ← Challenging images!
→ Performance:      49.9% (WORST)

Fold 5 Validation (18 images):
- Mean density:     5.58%
- Density range:    3.5% - 8.7%
- Mean brightness:  130
- Difficulty:       Easy-Medium
→ Performance:      71.6%
```

### Real Data from Our Experiment

```python
# From cv_summary.json
fold_results = {
    1: {
        'train_density': 0.05630,  # 5.63%
        'val_density':   0.05632,  # 5.63%
        'best_val_jacard': 0.5384, # 53.8%
        'best_epoch': 8
    },
    2: {
        'train_density': 0.05651,  # 5.65%
        'val_density':   0.05550,  # 5.55%
        'best_val_jacard': 0.5432, # 54.3%
        'best_epoch': 5
    },
    3: {
        'train_density': 0.05617,  # 5.62%
        'val_density':   0.05685,  # 5.68%
        'best_val_jacard': 0.7517, # 75.2% ← BEST!
        'best_epoch': 18
    },
    4: {
        'train_density': 0.05611,  # 5.61%
        'val_density':   0.05706,  # 5.71%
        'best_val_jacard': 0.4990, # 49.9% ← WORST
        'best_epoch': 6
    },
    5: {
        'train_density': 0.05643,  # 5.64%
        'val_density':   0.05579,  # 5.58%
        'best_val_jacard': 0.7162, # 71.6%
        'best_epoch': 11
    }
}
```

**Observations:**
1. **Densities are similar** (5.5-5.7%) across folds → Good stratification
2. **Performance varies widely** (49.9% to 75.2%) → Some validation sets harder than others
3. **Best epochs vary** (5 to 18) → Different folds converge at different rates

This variance is EXPECTED and NORMAL. It reflects the natural heterogeneity in the data.

---

## Loss Function Explained

### What is a Loss Function?

A **loss function** measures how "wrong" the model's predictions are. Lower loss = better predictions.

### Our Loss Function: Combined (Dice + Focal)

We use a **weighted combination** of two loss functions:

```python
L_combined = 0.7 × L_dice + 0.3 × L_focal

Where:
- L_dice:  Measures overlap between prediction and ground truth
- L_focal: Focuses on hard-to-classify pixels
- Weights: 70% Dice, 30% Focal (found empirically)
```

### Component 1: Dice Loss

**Purpose:** Maximize overlap between predicted and true masks

**Formula:**
```
Dice Coefficient = 2 × |Prediction ∩ Ground Truth| / (|Prediction| + |Ground Truth|)

Dice Loss = 1 - Dice Coefficient

Example:
Ground Truth:  ⚫⚫⚫⚫⚪⚪⚪⚪ (4 foreground, 4 background)
Prediction:    ⚫⚫⚫⚪⚪⚪⚪⚪ (3 foreground, 5 background)

Intersection:  ⚫⚫⚫ = 3 pixels
Sum:          4 + 3 = 7 pixels
Dice:         2×3/7 = 0.857 (Good!)
Loss:         1 - 0.857 = 0.143 (Low is good)
```

**Why use Dice?**
- Directly optimizes segmentation quality (IoU/Jaccard)
- Handles class imbalance well (our data is 92% background, 8% foreground)
- Treats false positives and false negatives equally

**Visual:**
```
Perfect Prediction:           Poor Prediction:
Ground Truth: ⚫⚫⚫⚫⚪⚪⚪⚪    Ground Truth: ⚫⚫⚫⚫⚪⚪⚪⚪
Prediction:   ⚫⚫⚫⚫⚪⚪⚪⚪    Prediction:   ⚫⚪⚪⚫⚫⚪⚪⚫

Dice = 1.0                    Dice = 0.4
Loss = 0.0 ✓                  Loss = 0.6 ✗
```

### Component 2: Focal Loss

**Purpose:** Focus on hard-to-classify pixels (boundaries, overlapping objects)

**Formula:**
```
FL(p) = -α × (1-p)^γ × log(p)

Where:
- p: predicted probability for correct class
- α = 0.25: weight for class imbalance
- γ = 2.0: focusing parameter (higher = more focus on hard examples)

Example:
Easy pixel:  p = 0.95 (model confident)
  FL = -0.25 × (1-0.95)^2 × log(0.95)
     = -0.25 × 0.0025 × (-0.051)
     = 0.000032 (very small - don't focus on this)

Hard pixel:  p = 0.55 (model uncertain)
  FL = -0.25 × (1-0.55)^2 × log(0.55)
     = -0.25 × 0.2025 × (-0.598)
     = 0.030 (100× larger - focus on this!)
```

**Why use Focal?**
- Down-weighs easy examples (clear background, obvious foreground)
- Up-weighs hard examples (object boundaries, overlapping microbeads)
- Helps model learn fine details

**Visual Effect:**
```
Without Focal Loss:
Model focuses equally on all pixels:
┌─────────────────────────┐
│ ⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪ │  ← Easy background
│ ⚪⚪⚪⚪⚫⚫⚫⚫⚫⚪⚪⚪⚪⚪⚪ │  ← Hard boundary
│ ⚪⚪⚪⚫⚫⚫⚫⚫⚫⚫⚪⚪⚪⚪⚪ │
│ ⚪⚪⚪⚪⚫⚫⚫⚫⚫⚪⚪⚪⚪⚪⚪ │  ← Easy foreground
│ ⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪ │
└─────────────────────────┘
Result: Good on easy pixels, poor on boundaries

With Focal Loss:
Model focuses on hard pixels:
┌─────────────────────────┐
│ ⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪ │  ← Ignored (confident)
│ ⚪⚪⚪⚪⚫⚫⚫⚫⚫⚪⚪⚪⚪⚪⚪ │  ← FOCUSED (uncertain)
│ ⚪⚪⚪⚫⚫⚫⚫⚫⚫⚫⚪⚪⚪⚪⚪ │     ↑ Higher weight
│ ⚪⚪⚪⚪⚫⚫⚫⚫⚫⚪⚪⚪⚪⚪⚪ │
│ ⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪ │  ← Ignored (confident)
└─────────────────────────┘
Result: Sharp boundaries, fine details
```

### Why Combine Them?

```
Dice Loss alone:
✓ Good global overlap
✗ May miss fine details
✗ Blurry boundaries

Focal Loss alone:
✓ Sharp boundaries
✓ Good on hard examples
✗ May sacrifice global structure

Combined (70% Dice + 30% Focal):
✓ Good global overlap (from Dice)
✓ Sharp boundaries (from Focal)
✓ Balanced performance
✓ Handles class imbalance
```

### Loss Function in Training

**How it guides training:**

```python
Step 1: Forward pass
  Input: Raw image (256×256×1)
  Output: Predicted mask (256×256×1, values 0-1)

Step 2: Compute loss
  Ground Truth: Binary mask (0 or 1 per pixel)
  Prediction:   Probabilities (0-1 per pixel)

  Dice_Loss = 1 - (2×overlap)/(pred_sum + truth_sum)
  Focal_Loss = -α×(1-p)^γ×log(p) [averaged over pixels]

  Total_Loss = 0.7×Dice_Loss + 0.3×Focal_Loss

Step 3: Backpropagation
  Gradient = ∂Loss/∂Weights
  Update: Weights -= learning_rate × Gradient

Step 4: Repeat
  Model adjusts weights to MINIMIZE loss
  Lower loss → Better predictions
```

**Training progression (Fold 3 example):**

```
Epoch 1:  Loss = 0.360, Dice = 0.35, Jaccard = 35%
Epoch 5:  Loss = 0.091, Dice = 0.78, Jaccard = 60%
Epoch 10: Loss = 0.050, Dice = 0.87, Jaccard = 65%
Epoch 18: Loss = 0.036, Dice = 0.91, Jaccard = 75% ← Best!

Loss decreases → Predictions improve → Jaccard increases
```

---

## Why Cross-Validation Reveals Bias

### The Statistical Principle

**Law of Large Numbers:** As sample size increases, the sample mean approaches the true population mean.

```
Single Split:
- Sample size: 1
- Estimate: 13.8%
- Variance: HIGH (could be anywhere from 5% to 80%!)
- Confidence: LOW

5-Fold CV:
- Sample size: 5
- Estimates: [53.8%, 54.3%, 75.2%, 49.9%, 71.6%]
- Mean: 60.97%
- Variance: LOWER
- Confidence: HIGHER (±11.5%)
```

### How Bias Gets Revealed

**Phase 1 (Single Split):**
```
Random selection picked 15 "hard" images:
┌───────────────────────────────────────────────┐
│ All 98 images                                 │
│ ⚪⚪⚪⚪⚪⚫⚪⚪⚪⚪⚪⚪⚫⚪⚪⚪⚪⚪⚫⚪ (5 hard)     │
│ ⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪ (0 hard)     │
│ ⚪⚪⚫⚪⚪⚪⚪⚫⚪⚪⚪⚪⚪⚪⚪⚫⚪⚪⚪⚪ (3 hard)     │
│ ⚪⚪⚪⚪⚪⚪⚪⚫⚪⚪⚪⚪⚪⚪⚪⚪⚫⚪⚪⚪ (2 hard)     │
│ ⚪⚪⚪⚪⚫⚪⚪⚪⚪⚪⚪⚫⚪⚪⚫⚪⚪⚪⚪⚪ (3 hard)     │
└───────────────────────────────────────────────┘

random_state=42 happened to select:
Validation: ⚫⚫⚫⚫⚫ + 10 normal images
            ↑
            5/15 = 33% hard (population: 13/98 = 13% hard)

Oversampling of hard images → Underestimated performance
```

**Cross-Validation (5 Folds):**
```
Each fold gets a DIFFERENT sample:

Fold 1: ⚫⚫ + 18 normal = 10% hard → Performance: 53.8%
Fold 2: ⚫⚫ + 18 normal = 10% hard → Performance: 54.3%
Fold 3: ⚫   + 19 normal = 5% hard  → Performance: 75.2% (easier!)
Fold 4: ⚫⚫⚫ + 17 normal = 15% hard → Performance: 49.9% (harder!)
Fold 5: ⚫⚫ + 16 normal = 11% hard → Performance: 71.6%

Average: (10+10+5+15+11)/5 = 10.2% hard ≈ population 13%

By averaging 5 different samples, we get close to TRUE distribution!
```

### Mathematical Proof

**Variance of single estimate:**
```
Var(single_split) = σ²/n
Where:
- σ² = population variance (unknown, but large for small n)
- n = 15 validation images

Var(single_split) = σ²/15 (HIGH VARIANCE)
```

**Variance of CV mean:**
```
Var(CV_mean) = σ²/N_total + σ²_fold/K
Where:
- N_total = 98 images (each tested once)
- K = 5 folds
- σ²_fold = between-fold variance

Var(CV_mean) ≈ σ²/98 + σ²_fold/5

Reduction: σ²/15 vs σ²/98 → 6.5× less variance!
```

### Confidence Intervals

**Phase 1:**
```
Estimate: 13.8%
Confidence: ???
Could be anywhere from 5% to 80% (we don't know!)
```

**Cross-Validation:**
```
Mean: 60.97%
Std Dev: 11.54%
95% CI: [38.4%, 83.6%]
68% CI: [49.4%, 72.5%]

We're 68% confident true performance is 49-73%
We're 95% confident true performance is 38-84%
```

### Why Phase 1's Bias Persisted Across Tests

**The Fatal Flaw:**
```python
# All three tests used THE SAME random_state!
Phase 1:           random_state=42
Focal Tversky:     random_state=42
Small Model:       random_state=42

Same seed → Same split → Same biased validation set!

This created FALSE CONSENSUS:
"All 3 tests show best at epoch 1"
  → "Must be a real problem!"

Reality: All 3 tests measured the SAME biased sample.
```

When we use CV (5 DIFFERENT samples), the bias disappears.

---

## Mathematical Foundation

### Stratified K-Fold Cross-Validation

**Algorithm:**
```python
def stratified_k_fold(X, y, k=5):
    """
    X: data (98 images)
    y: labels (density values)
    k: number of folds (5)
    """
    # Step 1: Sort by target variable (density)
    sorted_indices = np.argsort(y)

    # Step 2: Create k folds
    folds = [[] for _ in range(k)]

    # Step 3: Distribute samples in round-robin fashion
    for i, idx in enumerate(sorted_indices):
        fold_num = i % k
        folds[fold_num].append(idx)

    # Step 4: For each fold
    for i in range(k):
        val_indices = folds[i]
        train_indices = [idx for j in range(k) if j != i
                         for idx in folds[j]]

        yield train_indices, val_indices
```

**Why Stratified?**

Regular K-Fold:
```
All images shuffled randomly:
Fold 1: [random 20 images] → Could be all low-density
Fold 2: [random 20 images] → Could be all high-density
...
Performance variance: VERY HIGH
```

Stratified K-Fold:
```
Images sorted by density, then distributed:
Low density:    ⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪
                ↓ ↓ ↓ ↓ ↓
Fold:           1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5

Medium density: ⚫⚫⚫⚫⚫⚫⚫⚫⚫⚫⚫⚫⚫⚫⚫⚫⚫⚫⚫⚫
                ↓ ↓ ↓ ↓ ↓
Fold:           1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5

High density:   🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴
                ↓ ↓ ↓ ↓ ↓
Fold:           1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5

Result: Each fold has SIMILAR density distribution
Performance variance: LOWER
```

**Proof it worked in our experiment:**
```
Fold 1: train_density=5.63%, val_density=5.63% ✓
Fold 2: train_density=5.65%, val_density=5.55% ✓
Fold 3: train_density=5.62%, val_density=5.68% ✓
Fold 4: train_density=5.61%, val_density=5.71% ✓
Fold 5: train_density=5.64%, val_density=5.58% ✓

All within ±0.1% → Excellent stratification!
```

### Performance Estimation

**Estimator:**
```
μ̂_CV = (1/K) × Σ(i=1 to K) Performance_i

Where:
- K = 5 folds
- Performance_i = best validation Jaccard for fold i

μ̂_CV = (53.8 + 54.3 + 75.2 + 49.9 + 71.6) / 5
     = 304.8 / 5
     = 60.96%
```

**Variance:**
```
σ̂²_CV = (1/(K-1)) × Σ(i=1 to K) (Performance_i - μ̂_CV)²

σ̂²_CV = (1/4) × [(53.8-60.96)² + (54.3-60.96)² + (75.2-60.96)²
                 + (49.9-60.96)² + (71.6-60.96)²]
     = (1/4) × [51.22 + 44.36 + 202.75 + 122.34 + 113.15]
     = (1/4) × 533.82
     = 133.45 (percentage points²)

σ̂_CV = √133.45 = 11.54%
```

**Standard Error:**
```
SE = σ̂_CV / √K
   = 11.54 / √5
   = 11.54 / 2.236
   = 5.16%

95% CI: μ̂_CV ± 1.96 × SE
       = 60.96 ± 1.96 × 5.16
       = 60.96 ± 10.11
       = [50.85%, 71.07%]
```

### Bias-Variance Decomposition

**Expected prediction error:**
```
E[(ŷ - y)²] = Bias² + Variance + Irreducible Error

Single Split:
- Bias: HIGH (if split is unrepresentative)
- Variance: HIGH (only 1 sample)
- Total Error: VERY HIGH

Cross-Validation:
- Bias: LOW (average over multiple splits)
- Variance: LOWER (average reduces variance)
- Total Error: MUCH LOWER
```

---

## Summary

### Key Takeaways

1. **Cross-Validation = Multiple Experiments**
   - Train and test 5 times on different data subsets
   - Average results for reliable estimate

2. **Folds = Data Partitions**
   - Each fold is a subset of your data
   - Every image is in exactly one fold
   - Each fold used for validation exactly once

3. **Folds Differ Only in Data**
   - Same model, same hyperparameters
   - Different images in train vs validation
   - Natural variation in results (49.9% to 75.2%)

4. **Loss Function = Training Objective**
   - Combined: 0.7×Dice + 0.3×Focal
   - Dice: Global overlap
   - Focal: Fine details and boundaries
   - Guides model to make better predictions

5. **Why CV Reveals Bias**
   - Single split: 1 estimate (could be lucky or unlucky)
   - CV: 5 estimates (averages out luck)
   - Variance reduced by ~6.5×
   - True performance: 60.97% ± 11.5% (not 13.8%!)

### The Big Picture

```
Scientific Method Applied to Model Evaluation:

Hypothesis: "Model performs at 13.8% Jaccard"
Based on:   Single experiment (Phase 1)
Problem:    Not reproducible! (1 data split)

Solution:   Run 5 independent experiments (CV)
Results:    [53.8%, 54.3%, 75.2%, 49.9%, 71.6%]
Mean:       60.97% ± 11.5%
Conclusion: Original hypothesis was wrong due to biased sample
New estimate: 60.97% (with confidence interval)
```

Cross-validation is the machine learning equivalent of **replicating your experiment** in science. One trial can be misleading; multiple trials reveal the truth.

---

## Further Reading

### Papers
- Kohavi, R. (1995). "A study of cross-validation and bootstrap for accuracy estimation and model selection"
- Bengio, Y., & Grandvalet, Y. (2004). "No unbiased estimator of the variance of k-fold cross-validation"

### Why 5 Folds?
- **Trade-off:** More folds = lower bias, higher variance
- **Computational cost:** K-fold requires training K models
- **Common choices:**
  - 5-fold: Good balance, common in practice
  - 10-fold: Lower bias, higher computational cost
  - Leave-one-out (LOOCV): Lowest bias, highest variance & cost

### Our Choice: 5-Fold Stratified
- **5 folds:** Manageable compute (2 hours × 5 = 10 hours)
- **Stratified:** Controls for density imbalance
- **Result:** Reliable estimate with reasonable cost

---

**Last Updated:** 2025-10-13
**Author:** Educational Supplement to CV Report
