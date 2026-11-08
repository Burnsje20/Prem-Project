"""Probability metrics, calibration and saved chronological comparison tables."""
import numpy as np
from scipy.optimize import minimize_scalar


def probabilities(p):
    p = np.asarray(p,dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Expected finite Nx3 probabilities')
    if not np.allclose(p.sum(axis=1),1,atol=1e-6):
        raise ValueError('Probabilities do not sum to one')
    return p / p.sum(axis=1,keepdims=True)


def metrics(y, p):
    p, y = probabilities(p), np.asarray(y,dtype=int)
    if len(y) != len(p) or not len(y):
        raise ValueError('Empty or mismatched evaluation sample')
    onehot = np.eye(3)[y]
    confidence, correct = p.max(axis=1), (p.argmax(axis=1) == y)
    ece = 0.
    for lo in np.arange(0,1,.1):
        selected = (confidence >= lo) & (confidence < lo+.1 if lo < .9 else confidence <= 1.)
        if selected.any():
            ece += selected.mean()*abs(correct[selected].mean()-confidence[selected].mean())
    return {'matches':len(y),'log_loss':float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()),
            'brier':float(((p-onehot)**2).sum(axis=1).mean()),
            'accuracy':float(correct.mean()),'calibration_ece':float(ece)}


def reliability(y,p):
    p, y = probabilities(p), np.asarray(y)
    rows = []
    for outcome in range(3):
        for b in range(10):
            selected = (p[:,outcome] >= b/10) & (p[:,outcome] < (b+1)/10 if b < 9 else p[:,outcome] <= 1.)
            if selected.any():
                rows.append({'outcome':['H','D','A'][outcome], 'bin':b,
                             'count':int(selected.sum()),'mean_probability':float(p[selected,outcome].mean()),
                             'observed_frequency':float((y[selected] == outcome).mean())})
    return rows


def temperature(p, value):
    p = probabilities(p)
    logits = np.log(np.maximum(p,1e-15))/value
    logits -= logits.max(axis=1,keepdims=True)
    scaled = np.exp(logits)
    return scaled/scaled.sum(axis=1,keepdims=True)


def fit_temperature(y,p):
    fit = minimize_scalar(lambda t:metrics(y,temperature(p,t))['log_loss'],bounds=(.5,3),method='bounded')
    return float(fit.x)


def ensemble_weight(y,dc,xgb):
    rows = []
    for weight in np.linspace(0,1,21):
        row = metrics(y,(1-weight)*dc+weight*xgb)
        rows.append({'xgb_weight':float(weight),**row})
    return min(rows,key=lambda r:(r['log_loss'],r['brier']))['xgb_weight'], rows
