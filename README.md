# Drug Activity Prediction

A machine learning project to predict whether a molecule is active against the **BACE-1 target** using its chemical structure.

The BACE dataset from MoleculeNet is used to compare machine-learning approaches and molecular representations, and to assess how much activity information is available from molecular structure alone.

---

## Problem

This is a binary classification problem.

Each molecule has:

- A **SMILES** string describing its chemical structure
- An activity label for the BACE-1 target

The labels are:

```text
0 → Inactive
1 → Active
```

Here, "active" means that the compound was classified as active against the BACE-1 target in the dataset. It does not mean that the compound is an approved drug or that it has been shown to work in humans.

The prediction workflow is:

```text
Molecular structure
        ↓
Molecular features
        ↓
Machine learning model
        ↓
Probability of activity
```

---

## Dataset

The dataset contains **1,513 molecules**.

| Class | Number | Percentage |
|---|---:|---:|
| Inactive | 822 | 54.33% |
| Active | 691 | 45.67% |

### Sources

- MoleculeNet: https://moleculenet.org/
- DeepChem MoleculeNet documentation: https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html

---

## Tools used

- Python
- RDKit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Google Colab

---

## Approach

### 1. Reading and checking the data

The BACE dataset was loaded and checked for:

- dataset shape
- missing values
- activity distribution
- duplicate structures
- molecular validity
- basic molecular properties

These checks summarize the data before model training.

---

### 2. Converting SMILES into molecules

The structures were provided as SMILES strings.

**RDKit** converts the SMILES strings into molecular objects:

```text
SMILES
  ↓
RDKit molecule
```

The molecular objects support descriptor calculations and fingerprint generation.

---

## 3. Morgan fingerprints

The main models use **Morgan fingerprints** to represent each molecule.

The settings used were:

```text
Radius = 2
Number of bits = 2048
```

So every molecule was converted into a vector containing 2,048 binary features.

For example, conceptually:

```text
Molecule
   ↓
Morgan fingerprint
   ↓
[0, 1, 0, 0, 1, 0, ...]
```

The bits represent local structural environments in the molecule.

The final fingerprint representation had:

```text
1513 molecules × 2048 features
```

---

## 4. Molecular descriptors

The analysis also evaluates whether simpler molecular properties can predict activity.

Nine descriptors were calculated with RDKit:

- Molecular Weight
- LogP
- Hydrogen Bond Donors
- Hydrogen Bond Acceptors
- Rotatable Bonds
- Topological Polar Surface Area (TPSA)
- Ring Count
- Aromatic Ring Count
- Fraction CSP3

The two representations describe molecules at different levels:

```text
Morgan fingerprints
→ detailed structural information

Molecular descriptors
→ general physicochemical properties
```

---

## 5. Scaffold split

The data use a **scaffold-based split** rather than a random split.

This was done because a random split can put very similar molecules into both the training and test sets.

For example, two molecules might have the same core structure but slightly different substituents. If one is in training and the other is in testing, the test result can look better than the model's performance on genuinely new chemical structures.

With a scaffold split, molecules sharing the same core scaffold are kept together.

The data was divided into training, validation, and test sets based on these scaffolds.

This makes the evaluation more challenging and gives a better idea of how the model handles unseen chemical structures.

---

# Models

Three models were compared using Morgan fingerprints.

### Logistic Regression

Logistic Regression provides a simple baseline.

It is useful because it gives us a straightforward reference point before moving to more complex models.

### Random Forest

Random Forest combines predictions from many decision trees.

It can capture nonlinear relationships between molecular features, which is useful because molecular activity is unlikely to depend on each fingerprint bit independently.

### XGBoost

XGBoost is another tree-based method, but unlike Random Forest, the trees are built sequentially so that later trees can focus on correcting previous errors.

A second Random Forest, trained on the nine molecular descriptors, provides a comparison between the two molecular representations.

---

# Results

The models were evaluated on the held-out scaffold test set.

| Model | Features | Test ROC-AUC | Test PR-AUC |
|---|---|---:|---:|
| **Random Forest** | Morgan fingerprints | **0.9043** | **0.8908** |
| XGBoost | Morgan fingerprints | 0.8940 | 0.8885 |
| Logistic Regression | Morgan fingerprints | 0.8545 | 0.8366 |
| Random Forest | Molecular descriptors | 0.7949 | 0.7783 |

The Random Forest with Morgan fingerprints performed the best.

### Best result

```text
Model: Random Forest
Representation: 2048-bit Morgan fingerprint

Test ROC-AUC: 0.9043
Test PR-AUC:  0.8908
```

---

## Why ROC-AUC and PR-AUC?

ROC-AUC evaluates how well the model ranks active compounds above inactive compounds across probability thresholds, rather than measuring accuracy at a single threshold.

A ROC-AUC of:

```text
0.5 → roughly random
1.0 → perfect separation
```

The best model achieved a ROC-AUC of **0.9043**.

PR-AUC summarizes the precision-recall tradeoff for the active class.

The best model achieved a PR-AUC of **0.8908**.

---

# Fingerprints vs. Molecular Descriptors

The two molecular representations produced different results.

The Random Forest using Morgan fingerprints achieved:

```text
ROC-AUC = 0.9043
```

while the Random Forest using the nine molecular descriptors achieved:

```text
ROC-AUC = 0.7949
```

For this prediction task, Morgan fingerprints provided more useful information than the selected global molecular properties.

Molecular weight, LogP, and TPSA alone did not capture all of the predictive information available from molecular structure.

---

# Model Interpretation

After model comparison, feature importance was examined to see which fingerprint bits influenced the Random Forest.

## Feature importance

Random Forest provides feature importance scores for the fingerprint features.

This identifies the fingerprint bits with the largest influence on the model.

A fingerprint bit does not directly mean something simple like "contains oxygen". It represents a hashed local structural environment.

RDKit was used to trace important fingerprint bits back to molecular environments where possible.

---

## SHAP

**SHAP (SHapley Additive exPlanations)** provides a more detailed view of the model's predictions.

SHAP helps answer questions such as:

```text
Which features are pushing a prediction toward active?
Which features are pushing it toward inactive?
```

This is useful for understanding the model rather than treating it as a complete black box.

One important limitation is that SHAP or feature importance does **not** prove that a particular chemical substructure causes activity. It only tells us that the model found that feature useful for making predictions from this dataset.

---

# Predicting a New Molecule

A prediction function applies the trained model to a new SMILES string.

The workflow is:

```text
New SMILES
    ↓
RDKit
    ↓
Morgan fingerprint
    ↓
Trained Random Forest
    ↓
Activity probability
    ↓
Active / Inactive
```

For example:

```python
prediction, probability = predict_activity(smiles)

print(prediction)
print(probability)
```

The output gives both the predicted class and the model's estimated probability.

These predictions should be treated as a way to **prioritize compounds for further investigation**, not as a replacement for experimental testing.

---

# Key observations

Key observations:

- Molecular representation made a large difference in model performance.
- Morgan fingerprints worked substantially better than the selected descriptor set.
- Random Forest performed slightly better than XGBoost on this scaffold-based test set.
- Scaffold splitting makes the problem harder than a simple random split, but gives a more realistic test of generalization to new chemical structures.
- Model interpretation is useful, but important features should not automatically be treated as causal biological mechanisms.

---

# Limitations

There are still several things that could be improved.

- The results are based on one scaffold-based split.
- The descriptor set is relatively small.
- Hyperparameter tuning was limited.
- Morgan fingerprints can have collisions because different molecular environments can map to the same bit.
- There was no independent external dataset used for final validation.
- Computational predictions still need experimental validation.

Because of these limitations, a ROC-AUC of 0.9043 does not imply equal performance on every new chemical library.

---

# Future Improvements

Potential next steps:

- Repeated scaffold cross-validation
- More systematic hyperparameter tuning
- External dataset validation
- Additional molecular fingerprints
- More molecular descriptors
- Probability calibration
- Uncertainty estimation
- Threshold optimization
- Ensemble models
- Graph neural networks
- More detailed chemical interpretation

---

# Final Result

The best-performing model was:

```text
Random Forest
+
2048-bit Morgan fingerprints
```

with:

```text
Test ROC-AUC = 0.9043
Test PR-AUC  = 0.8908
```

The workflow covers molecular structure processing, feature generation, scaffold-based evaluation, model comparison, prediction interpretation, and activity prediction for held-out molecules.
