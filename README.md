# Drug Activity Prediction

A machine learning project to predict whether a molecule is active against the **BACE-1 target** using its chemical structure.

I used the BACE dataset from MoleculeNet and compared a few different machine learning approaches and molecular representations. The main idea was to see how much useful information we can get from the structure of a molecule alone.

---

## What is the problem?

This is a binary classification problem.

For each molecule, we have:

- A **SMILES** string describing its chemical structure
- An activity label for the BACE-1 target

The labels are:

```text
0 → Inactive
1 → Active
```

Here, "active" means that the compound was classified as active against the BACE-1 target in the dataset. It does not mean that the compound is an approved drug or that it has been shown to work in humans.

The goal of the project is therefore:

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

I first loaded the BACE dataset and checked:

- dataset shape
- missing values
- activity distribution
- duplicate structures
- molecular validity
- basic molecular properties

This was mainly to understand what I was working with before training any models.

---

### 2. Converting SMILES into molecules

The structures were provided as SMILES strings.

I used **RDKit** to convert them into molecular objects:

```text
SMILES
  ↓
RDKit molecule
```

This allowed me to calculate molecular properties and generate fingerprints from the structures.

---

## 3. Morgan fingerprints

For the main ML models, I represented each molecule using a **Morgan fingerprint**.

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

I also wanted to see whether simpler molecular properties could predict activity.

I calculated nine descriptors using RDKit:

- Molecular Weight
- LogP
- Hydrogen Bond Donors
- Hydrogen Bond Acceptors
- Rotatable Bonds
- Topological Polar Surface Area (TPSA)
- Ring Count
- Aromatic Ring Count
- Fraction CSP3

This gave me two different ways of describing a molecule:

```text
Morgan fingerprints
→ detailed structural information

Molecular descriptors
→ general physicochemical properties
```

---

## 5. Scaffold split

Instead of simply randomly splitting the molecules, I used a **scaffold-based split**.

This was done because a random split can put very similar molecules into both the training and test sets.

For example, two molecules might have the same core structure but slightly different substituents. If one is in training and the other is in testing, the test result can look better than the model's performance on genuinely new chemical structures.

With a scaffold split, molecules sharing the same core scaffold are kept together.

The data was divided into training, validation, and test sets based on these scaffolds.

This makes the evaluation more challenging and gives a better idea of how the model handles unseen chemical structures.

---

# Models

I compared three models using Morgan fingerprints.

### Logistic Regression

I used Logistic Regression as a simple baseline.

It is useful because it gives us a straightforward reference point before moving to more complex models.

### Random Forest

Random Forest combines predictions from many decision trees.

It can capture nonlinear relationships between molecular features, which is useful because molecular activity is unlikely to depend on each fingerprint bit independently.

### XGBoost

XGBoost is another tree-based method, but unlike Random Forest, the trees are built sequentially so that later trees can focus on correcting previous errors.

I also trained a Random Forest using the nine molecular descriptors to compare the two molecular representations.

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

I used ROC-AUC rather than relying only on accuracy because the model produces probabilities and we want to evaluate how well it separates active compounds from inactive compounds across different thresholds.

A ROC-AUC of:

```text
0.5 → roughly random
1.0 → perfect separation
```

The best model achieved a ROC-AUC of **0.9043**.

I also used PR-AUC because it focuses more directly on the precision-recall tradeoff for the active class.

The best model achieved a PR-AUC of **0.8908**.

---

# Fingerprints vs. Molecular Descriptors

One of the more interesting results was the difference between the two molecular representations.

The Random Forest using Morgan fingerprints achieved:

```text
ROC-AUC = 0.9043
```

while the Random Forest using the nine molecular descriptors achieved:

```text
ROC-AUC = 0.7949
```

This suggests that the detailed structural information captured by the Morgan fingerprints was more useful for this particular prediction task than the selected global molecular properties.

In other words, properties such as molecular weight, LogP and TPSA alone were not enough to capture all the information the model could learn from the molecular structure.

---

# Model Interpretation

After comparing the models, I also looked at what the Random Forest was actually using.

## Feature importance

Random Forest provides feature importance scores for the fingerprint features.

This helped identify which of the 2,048 fingerprint bits had the largest influence on the model.

A fingerprint bit does not directly mean something simple like "contains oxygen". It represents a hashed local structural environment.

So I used RDKit to trace important fingerprint bits back to molecular environments where possible.

---

## SHAP

I also used **SHAP (SHapley Additive exPlanations)** to get a more detailed view of the model's predictions.

SHAP helps answer questions such as:

```text
Which features are pushing a prediction toward active?
Which features are pushing it toward inactive?
```

This is useful for understanding the model rather than treating it as a complete black box.

One important limitation is that SHAP or feature importance does **not** prove that a particular chemical substructure causes activity. It only tells us that the model found that feature useful for making predictions from this dataset.

---

# Predicting a New Molecule

I also created a small prediction function so that the trained model can be used with a new SMILES string.

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

# What I learned

A few things stood out from the project:

1. Molecular representation made a large difference in model performance.
2. Morgan fingerprints worked substantially better than the small descriptor set I tested.
3. Random Forest performed slightly better than XGBoost on this scaffold-based test set.
4. Scaffold splitting makes the problem harder than a simple random split, but gives a more realistic test of generalization to new chemical structures.
5. Model interpretation is useful, but important features should not automatically be treated as causal biological mechanisms.

---

# Limitations

There are still several things that could be improved.

- The results are based on one scaffold-based split.
- The descriptor set is relatively small.
- Hyperparameter tuning was limited.
- Morgan fingerprints can have collisions because different molecular environments can map to the same bit.
- There was no independent external dataset used for final validation.
- Computational predictions still need experimental validation.

Because of these limitations, I would not claim that a ROC-AUC of 0.9043 means the model will perform equally well on every new chemical library.

---

# Future Improvements

Some things I would like to try next:

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

# Project Structure

```text
drug-activity-prediction/
│
├── README.md
├── Drug_Activity_Prediction.ipynb
│
├── data/
│   └── README.md
│
├── figures/
│   ├── class_distribution.png
│   ├── model_comparison.png
│   ├── representation_comparison.png
│   ├── shap_importance.png
│   └── prediction_distribution.png
│
└── requirements.txt
```

---

# Final Result

The final model from this project was:

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

Overall, this project gave me experience with the complete workflow of a small molecular machine-learning problem: starting from chemical structures, choosing a molecular representation, creating a scaffold-based evaluation split, comparing different ML models, interpreting the predictions, and finally using the trained model to predict activity for unseen molecules.
