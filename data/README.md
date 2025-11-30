# Data Directory

## Overview
This directory contains financial transaction data used for fraud detection model training and evaluation.

## Dataset Information

### Data Source
- **Source**: Credit card transaction datasets (e.g., Kaggle Fraud Dataset)
- **Type**: Anonymized credit card transactions
- **Features**: 30 principal components + transaction amount
- **Target Variable**: Binary classification (fraud/legitimate)

### Data Structure
```
data/
├── raw/
│   ├── creditcard_transactions.csv
│   └── fraud_labels.csv
├── processed/
│   ├── train_data.csv
│   ├── test_data.csv
│   └── validation_data.csv
└── results/
    └── predictions.csv
```

## Data Characteristics
- **Total Records**: Varies by dataset
- **Fraud Rate**: Highly imbalanced (~0.1% fraud)
- **Features**: 30 PCA-transformed features + time + amount
- **Target Classes**: 0 (Legitimate), 1 (Fraudulent)

## Class Imbalance
The dataset is highly imbalanced. SMOTE (Synthetic Minority Over-sampling Technique) is used during preprocessing to handle this.

## Data Preprocessing
1. Normalization: StandardScaler for features
2. Feature scaling: Amount feature scaled separately
3. SMOTE: Applied to training data only
4. Train-Test Split: 70-30 with stratification

## Usage
The `FraudDetector` class loads and preprocesses data from this directory.

## Licensing & Privacy
All datasets are anonymized. Refer to the original source repository for licensing terms.
