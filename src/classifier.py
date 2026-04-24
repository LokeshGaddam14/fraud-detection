"""
Fraud Detection Classifier — Production Grade
IEEE-CIS Transaction Dataset | XGBoost + LightGBM Ensemble
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix,
    recall_score, precision_score, f1_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import os


class FraudDetector:
    """
    Production-grade fraud detection classifier.

    Supports XGBoost and LightGBM with:
    - SMOTE class imbalance handling (training only)
    - Business-cost-aware threshold optimization
    - SHAP-ready predict_proba interface
    - Consistent train/val/test evaluation
    """

    # Business cost assumptions (USD)
    COST_FN    = 400   # cost of missing one fraud
    COST_FP    = 8     # cost of blocking a legitimate transaction
    REVENUE_TN = 2     # interchange revenue per approved legit transaction

    def __init__(self, model_type: str = 'xgboost'):
        assert model_type in ('xgboost', 'lightgbm'), \
            "model_type must be 'xgboost' or 'lightgbm'"
        self.model_type     = model_type
        self.model          = None
        self.scaler         = StandardScaler()
        self.smote          = SMOTE(sampling_strategy=0.20, k_neighbors=5, random_state=42)
        self.optimal_threshold = 0.5
        self.feature_names  = None

    # ------------------------------------------------------------------ #
    #  MODEL BUILD                                                         #
    # ------------------------------------------------------------------ #
    def build_model(self):
        """Instantiate model with tuned hyperparameters."""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators      = 500,
                max_depth         = 7,
                learning_rate     = 0.05,
                subsample         = 0.80,
                colsample_bytree  = 0.80,
                min_child_weight  = 5,
                gamma             = 0.1,
                eval_metric       = 'auc',
                use_label_encoder = False,
                random_state      = 42
            )
        else:
            self.model = lgb.LGBMClassifier(
                n_estimators     = 500,
                max_depth        = 7,
                learning_rate    = 0.05,
                subsample        = 0.80,
                colsample_bytree = 0.80,
                min_child_samples= 20,
                random_state     = 42,
                verbose          = -1
            )
        return self.model

    # ------------------------------------------------------------------ #
    #  PREPROCESSING                                                       #
    # ------------------------------------------------------------------ #
    def preprocess(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Scale features. fit=True on training set only."""
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def apply_smote(self, X: np.ndarray, y: pd.Series):
        """
        Apply SMOTE to training data ONLY.
        Never apply to validation or test sets.
        """
        X_bal, y_bal = self.smote.fit_resample(X, y)
        print(f"  SMOTE | Before: {y.mean()*100:.2f}% fraud → After: {y_bal.mean()*100:.2f}% fraud")
        return X_bal, y_bal

    # ------------------------------------------------------------------ #
    #  TRAINING                                                            #
    # ------------------------------------------------------------------ #
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Full training pipeline: scale → SMOTE → fit."""
        self.feature_names = list(X_train.columns)
        self.build_model()

        X_sc           = self.preprocess(X_train, fit=True)
        X_bal, y_bal   = self.apply_smote(X_sc, y_train)

        if X_val is not None and self.model_type == 'xgboost':
            X_val_sc = self.preprocess(X_val, fit=False)
            self.model.set_params(early_stopping_rounds=50, n_estimators=1000)
            self.model.fit(
                X_bal, y_bal,
                eval_set=[(X_val_sc, y_val)],
                verbose=100
            )
        else:
            self.model.fit(X_bal, y_bal)

        print(f"  Training complete | Model: {self.model_type}")
        return self

    # ------------------------------------------------------------------ #
    #  INFERENCE                                                           #
    # ------------------------------------------------------------------ #
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probability scores."""
        X_sc = self.preprocess(X, fit=False)
        return self.model.predict_proba(X_sc)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """Binary prediction at given threshold (default: optimal_threshold)."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        return (self.predict_proba(X) >= thresh).astype(int)

    # ------------------------------------------------------------------ #
    #  THRESHOLD OPTIMIZATION                                              #
    # ------------------------------------------------------------------ #
    def optimize_threshold(self, X_val: pd.DataFrame, y_val: pd.Series,
                           target_recall: float = 0.90) -> float:
        """
        Find threshold that maximizes precision while achieving target_recall.
        Business logic: prioritize catching fraud (recall) over minimizing false alarms.
        """
        probs = self.predict_proba(X_val)
        precisions, recalls, thresholds = precision_recall_curve(y_val, probs)

        valid = np.where(recalls[:-1] >= target_recall)[0]
        if len(valid) > 0:
            best_idx = valid[np.argmax(precisions[valid])]
            self.optimal_threshold = float(thresholds[best_idx])
        else:
            print(f"  Warning: Cannot achieve {target_recall*100:.0f}% recall. Using 0.5.")
            self.optimal_threshold = 0.5

        print(f"  Optimal threshold: {self.optimal_threshold:.3f} "
              f"(recall≥{target_recall*100:.0f}%)")
        return self.optimal_threshold

    # ------------------------------------------------------------------ #
    #  EVALUATION                                                          #
    # ------------------------------------------------------------------ #
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                 threshold: float = None, label: str = 'Test') -> dict:
        """Full evaluation: AUC, recall, precision, F1, business impact."""
        probs  = self.predict_proba(X_test)
        thresh = threshold if threshold is not None else self.optimal_threshold
        preds  = (probs >= thresh).astype(int)

        roc_auc = roc_auc_score(y_test, probs)
        pr_auc  = average_precision_score(y_test, probs)
        rec     = recall_score(y_test, preds)
        prec    = precision_score(y_test, preds, zero_division=0)
        f1      = f1_score(y_test, preds)

        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        net_benefit = (tp * self.COST_FN) - (fn * self.COST_FN) - (fp * self.COST_FP)

        results = {
            'label'         : label,
            'threshold'     : thresh,
            'roc_auc'       : round(roc_auc, 4),
            'pr_auc'        : round(pr_auc, 4),
            'recall'        : round(rec, 4),
            'precision'     : round(prec, 4),
            'f1_score'      : round(f1, 4),
            'tp'            : int(tp),
            'fp'            : int(fp),
            'fn'            : int(fn),
            'tn'            : int(tn),
            'net_benefit_usd': int(net_benefit)
        }

        print(f"\n{'='*55}")
        print(f"  EVALUATION — {label} | Threshold={thresh:.3f}")
        print(f"{'='*55}")
        print(f"  ROC-AUC    : {roc_auc:.4f}")
        print(f"  PR-AUC     : {pr_auc:.4f}")
        print(f"  Recall     : {rec:.4f}   (fraud caught)")
        print(f"  Precision  : {prec:.4f}")
        print(f"  F1-Score   : {f1:.4f}")
        print(f"  TP={tp:,} | FP={fp:,} | FN={fn:,} | TN={tn:,}")
        print(f"  Net Benefit: ${net_benefit:,.0f}")

        return results

    # ------------------------------------------------------------------ #
    #  PERSISTENCE                                                         #
    # ------------------------------------------------------------------ #
    def save(self, output_dir: str = '../outputs/models'):
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.model,  f'{output_dir}/model.pkl')
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        meta = {
            'model_type'        : self.model_type,
            'optimal_threshold' : self.optimal_threshold,
            'feature_names'     : self.feature_names
        }
        with open(f'{output_dir}/meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  Model saved to {output_dir}/")

    def load(self, output_dir: str = '../outputs/models'):
        self.model  = joblib.load(f'{output_dir}/model.pkl')
        self.scaler = joblib.load(f'{output_dir}/scaler.pkl')
        with open(f'{output_dir}/meta.json') as f:
            meta = json.load(f)
        self.model_type         = meta['model_type']
        self.optimal_threshold  = meta['optimal_threshold']
        self.feature_names      = meta['feature_names']
        print(f"  Model loaded | threshold={self.optimal_threshold:.3f}")
        return self
