"""Time-decayed, regularized Dixon-Coles likelihood and coherent score markets."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import poisson

from .data import clubs, utc


def tau(home, away, lam, mu, rho):
    return np.select([(home == 0) & (away == 0), (home == 0) & (away == 1),
                      (home == 1) & (away == 0), (home == 1) & (away == 1)],
                     [1 - lam * mu * rho, 1 + lam * rho, 1 + mu * rho, 1 - rho], default=1.)


def score_matrix(lam, mu, rho=0., min_goals=0):
    if not np.isfinite([lam, mu, rho]).all() or lam <= 0 or mu <= 0:
        raise ValueError('Invalid expected goals or rho')
    limit = max(2, min_goals, int(max(poisson.ppf(1 - 1e-12, lam), poisson.ppf(1 - 1e-12, mu))))
    h, a = np.indices((limit + 1, limit + 1))
    correction = tau(h, a, lam, mu, rho)
    if correction.min() <= 0:
        raise ValueError('Dixon-Coles correction must remain positive')
    matrix = poisson.pmf(h, lam) * poisson.pmf(a, mu) * correction
    return matrix / matrix.sum()


def outcomes(matrix):
    return np.array([np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()])


def reconcile(matrix, probabilities):
    """Minimum-KL adjustment with H/D/A mass constraints; conditional scores retained."""
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (3,) or (p < 0).any() or not np.isfinite(p).all() or not np.isclose(p.sum(), 1):
        raise ValueError('Invalid outcome probabilities')
    h, a = np.indices(matrix.shape)
    result = matrix.copy()
    for mask, target in zip([h > a, h == a, h < a], p):
        mass = result[mask].sum()
        if mass <= 0:
            raise ValueError('Empty score region')
        result[mask] *= target / mass
    return result / result.sum()


def markets(matrix):
    h, a = np.indices(matrix.shape)
    indices = np.argsort(-matrix.ravel(), kind='stable')[:5]
    top = [{'score': f'{i // matrix.shape[1]}-{i % matrix.shape[1]}',
            'probability': float(matrix.ravel()[i])} for i in indices]
    p = outcomes(matrix)
    return {'home_win_probability': float(p[0]), 'draw_probability': float(p[1]),
            'away_win_probability': float(p[2]), 'most_likely_outcome': ['H','D','A'][int(p.argmax())],
            'expected_home_goals': float((h * matrix).sum()),
            'expected_away_goals': float((a * matrix).sum()),
            'most_likely_score': top[0]['score'], 'top_five_scorelines': top,
            'both_teams_to_score': float(matrix[(h > 0) & (a > 0)].sum()),
            'over_2_5': float(matrix[h + a >= 3].sum()),
            'under_2_5': float(matrix[h + a <= 2].sum())}


class DixonColes:
    def __init__(self, half_life=365, prior_weight=8, independent=False):
        self.half_life, self.prior_weight, self.independent = half_life, prior_weight, independent

    def fit(self, matches, cutoff, prior=None, extra_teams=()):
        self.cutoff = utc(cutoff).normalize()
        data = matches[matches.available_at <= self.cutoff].copy()
        if len(data) < 10:
            raise ValueError('At least ten completed matches are required')
        self.teams = sorted(clubs(data) | set(extra_teams))
        self.lookup = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        centres = np.zeros((n, 2))
        promoted = set() if prior is None else prior.promoted & set(self.teams)
        for team in promoted:
            centres[self.lookup[team]] = prior.predict(team)
        # Remove stale prior-PL evidence for re-promoted clubs, retaining their
        # new PL matches and historical opponents' matches against other teams.
        if prior is not None:
            old = data.season < prior.target_season
            data = data[~(old & (data.home_team.isin(promoted) | data.away_team.isin(promoted)))]
        hi = data.home_team.map(self.lookup).to_numpy()
        ai = data.away_team.map(self.lookup).to_numpy()
        hg, ag = data.home_goals.to_numpy(), data.away_goals.to_numpy()
        ages = (self.cutoff - data.date).dt.total_seconds().to_numpy() / 86400
        weights = np.exp(-np.log(2) * ages / self.half_life)
        regularization = np.full(n, 2.)
        for team in promoted:
            regularization[self.lookup[team]] = self.prior_weight

        def unpack(theta):
            # Two sum-to-zero constraints identify the intercept and team effects.
            attack, defence = theta[2:2+n], theta[2+n:2+2*n]
            attack, defence = attack - attack.mean(), defence - defence.mean()
            # Include neutral unknown-team fallback in the admissible rho range.
            max_a, min_d = max(0., attack.max()), min(0., defence.min())
            lmax = np.exp(theta[0] + theta[1] + max_a - min_d)
            mmax = np.exp(theta[0] + max_a - min_d)
            lower = -1 / max(lmax, mmax)
            upper = min(1., 1 / (lmax * mmax))
            rho = 0. if self.independent else lower + (upper-lower) * expit(theta[-1])
            # Keep strictly inside the valid interval for all team pairings.
            rho *= (1 - 1e-6)
            return attack, defence, rho

        def objective(theta):
            attack, defence, rho = unpack(theta)
            log_l = theta[0] + theta[1] + attack[hi] - defence[ai]
            log_m = theta[0] + attack[ai] - defence[hi]
            lam, mu = np.exp(log_l), np.exp(log_m)
            correction = tau(hg, ag, lam, mu, rho)
            ll = hg * log_l - lam - gammaln(hg+1) + ag * log_m - mu - gammaln(ag+1)
            penalty = .5 * np.sum(regularization[:,None] * (np.column_stack([attack,defence]) - centres)**2)
            return float(-np.sum(weights * (ll + np.log(correction))) + penalty)

        initial = np.zeros(3 + 2*n)
        initial[0] = np.log(max(.1, data.away_goals.mean()))
        initial[1] = np.log(max(.1, data.home_goals.mean()) / max(.1, data.away_goals.mean()))
        fit = minimize(objective, initial, method='L-BFGS-B',
                       bounds=[(-2,2),(-1,1)] + [(-2,2)]*(2*n) + [(-10,10)],
                       options={'maxiter': 350, 'maxfun': 60000, 'ftol': 1e-8})
        if not fit.success:
            raise RuntimeError(f'Dixon-Coles optimization failed: {fit.message}')
        self.attack, self.defence, self.rho = unpack(fit.x)
        self.intercept, self.home_advantage = map(float, fit.x[:2])
        self.promoted = sorted(promoted)
        self.diagnostics = {'converged': True, 'objective': float(fit.fun), 'iterations': int(fit.nit),
                            'training_matches': len(data), 'effective_matches': float(weights.sum()),
                            'promotion_samples': len(prior.samples) if prior else 0}
        return self

    def rates(self, home, away):
        if home == away:
            raise ValueError('Choose two different teams')
        def rating(team):
            i = self.lookup.get(team)
            return (0.,0.) if i is None else (self.attack[i], self.defence[i])
        ha, hd = rating(home)
        aa, ad = rating(away)
        return np.exp(self.intercept + self.home_advantage + ha - ad), np.exp(self.intercept + aa - hd)

    def matrix(self, home, away, min_goals=0):
        lam, mu = self.rates(home, away)
        return score_matrix(lam, mu, self.rho, min_goals)

    def save(self, path):
        payload = {'half_life': self.half_life, 'prior_weight': self.prior_weight, 'independent': self.independent,
                   'cutoff': self.cutoff.isoformat(), 'teams': self.teams,
                   'attack': self.attack.tolist(), 'defence': self.defence.tolist(),
                   'rho': float(self.rho), 'intercept': self.intercept,
                   'home_advantage': self.home_advantage, 'promoted': self.promoted,
                   'diagnostics': self.diagnostics}
        Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text(encoding='utf-8'))
        model = cls(obj.pop('half_life'), obj.pop('prior_weight'), obj.pop('independent'))
        model.__dict__.update(obj)
        model.cutoff = utc(model.cutoff)
        model.attack, model.defence = np.array(model.attack), np.array(model.defence)
        model.lookup = {t:i for i,t in enumerate(model.teams)}
        return model
