"""Fraud Detection Classifier

Real-time machine learning system for detecting fraudulent transactions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import joblib


class FraudDetector:
    """Machine learning classifier for fraud detection."""
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
    
    def build_model(self):
        """Build the fraud detection model."""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=1,
                random_state=42
            )
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42
            )
        return self.model
    
    def handle_imbalance(self, X, y):
        """Handle class imbalance using SMOTE."""
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def preprocess_data(self, X, fit=True):
        """Preprocess transaction features."""
        X_processed = X.copy()
        if fit:
            X_processed = self.scaler.fit_transform(X_processed)
        else:
            X_processed = self.scaler.transform(X_processed)
        return X_processed
    
    def train(self, X_train, y_train):
        """Train model with imbalance handling."""
        X_processed = self.preprocess_data(X_train, fit=True)
        X_balanced, y_balanced = self.handle_imbalance(X_processed, y_train)
        self.model.fit(X_balanced, y_balanced)
    
    def predict(self, X):
        """Predict fraud probability."""
        X_processed = self.preprocess_data(X, fit=False)
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        return predictions, probabilities
    
    def save(self, filepath):
        """Save trained model."""
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        """Load pre-trained model."""
        self.model = joblib.load(filepath)
