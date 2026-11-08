"""Small-data XGBoost configuration, ordered schemas and native persistence."""
import numpy as np
from xgboost import XGBClassifier

from .evaluation import probabilities


def train_booster(train, columns, params, seed=42):
    if set(train.target.unique()) != {0,1,2}:
        raise ValueError('Training requires all H/D/A classes')
    model = XGBClassifier(objective='multi:softprob',num_class=3,eval_metric='mlogloss',
                          random_state=seed,n_jobs=2,tree_method='hist',**params)
    model.fit(train[columns].astype(float),train.target.to_numpy())
    return model


def predict_booster(model, frame, columns):
    missing = set(columns)-set(frame)
    if missing:
        raise ValueError(f'Feature schema mismatch: {sorted(missing)}')
    return probabilities(model.predict_proba(frame[columns].astype(float)))


def load_booster(path):
    model = XGBClassifier()
    model.load_model(path)
    return model
