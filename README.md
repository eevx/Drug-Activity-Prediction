# \# Drug Activity Prediction using Machine Learning

# 

# Predicting whether small molecules are active against the \*\*BACE-1 biological target\*\* using molecular fingerprints, physicochemical descriptors, and classical machine-learning models.

# 

# \## Overview

# 

# This project develops a machine-learning pipeline for binary classification of molecular activity against the BACE-1 target.

# 

# The main objective is to investigate whether molecular structure can be used to distinguish \*\*active\*\* and \*\*inactive\*\* compounds and to compare different molecular representations and machine-learning algorithms.

# 

# The project focuses on:

# 

# \- Molecular representation using RDKit

# \- Morgan molecular fingerprints

# \- Physicochemical molecular descriptors

# \- Scaffold-based dataset splitting

# \- Logistic Regression, Random Forest, and XGBoost

# \- ROC-AUC and PR-AUC evaluation

# \- Feature importance and SHAP-based interpretation

# \- Prediction of activity for unseen molecules

# 

# \---

# 

# \## Dataset

# 

# The project uses the \*\*BACE dataset\*\* from the MoleculeNet benchmark.

# 

# The dataset contains molecular structures represented as SMILES strings together with experimentally determined activity labels for the BACE-1 target.

# 

# \### Dataset statistics

# 

# | Property | Value |

# |---|---:|

# | Total molecules | 1,513 |

# | Active | 691 |

# | Inactive | 822 |

# | Active percentage | 45.67% |

# | Inactive percentage | 54.33% |

# 

# \### Data source

# 

# \- MoleculeNet: https://moleculenet.org/

# \- BACE dataset: https://deepchem.readthedocs.io/en/latest/api\_reference/moleculenet.html

# 

# \---

# 

# \## Problem Formulation

# 

# This is a \*\*binary classification problem\*\*.

# 

# Given a molecule:

# 

# ```text

# SMILES → Molecular representation → ML model → P(Active)

