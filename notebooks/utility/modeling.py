from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassifierMixin,
    clone
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    cohen_kappa_score,
)
from lightgbm import Dataset as LGBMDataSet
from lightgbm import early_stopping, cv, LGBMClassifier
import optuna
from optuna.samplers import TPESampler
import numpy as np


class TargetTransformer(BaseEstimator, TransformerMixin):
    def fit(self, y):
        self.min = y.min()
        return self

    def transform(self, y):
        return y - self.min

    def inverse_transform(self, y):
        return y + self.min


class CustomTransformedTargetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier, transformer):
        self.classifier = classifier
        self.transformer = transformer

    def fit(self, X, y):
        y_transformed = self.transformer.fit_transform(y)
        self.classifier_ = clone(self.classifier)
        self.classifier_.fit(X, y_transformed)
        return self

    def predict(self, X):
        y_pred_transformed = self.classifier_.predict(X)
        return self.transformer.inverse_transform(y_pred_transformed)

    def predict_proba(self, X):
        return self.classifier_.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        y_predicted = self.predict(X)
        return ({
            "Accuracy": accuracy_score(y_true=y, y_pred=y_predicted),
            "Balanced Accuracy": balanced_accuracy_score(y_true=y, y_pred=y_predicted),
            "Weighted F1": f1_score(y_true=y, y_pred=y_predicted, average="weighted"),
            "Weighted Recall": recall_score(y_true=y, y_pred=y_predicted, average="weighted"),
            "QWK": cohen_kappa_score(y1=y, y2=y_predicted, weights="quadratic"),
        })