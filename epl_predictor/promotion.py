"""Regularized Championship-to-PL conversions using completed transitions."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .data import clubs, season_before


def strengths(frame):
    """Log scoring/conceding strengths relative to the division scoring environment."""
    if frame.empty:
        return {}
    baseline = (frame.home_goals.sum() + frame.away_goals.sum()) / (2 * len(frame))
    result = {}
    for team in sorted(clubs(frame)):
        home, away = frame[frame.home_team == team], frame[frame.away_team == team]
        n = len(home) + len(away)
        # Half-goal smoothing prevents log(0) for small synthetic/partial samples.
        gf = (home.home_goals.sum() + away.away_goals.sum() + .5) / (n + .5 / baseline)
        ga = (home.away_goals.sum() + away.home_goals.sum() + .5) / (n + .5 / baseline)
        result[team] = [float(np.log(gf / baseline)), float(-np.log(ga / baseline))]
    return result


class PromotionPrior:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, pl, championship, target_season, target_teams=None):
        self.target_season = target_season
        self.samples = []
        for season in sorted(s for s in set(pl.season) if s < target_season):
            prev = season_before(season)
            prior_pl = pl[pl.season == prev]
            current_pl = pl[pl.season == season]
            prior_ch = championship[championship.season == prev]
            # Complete league seasons only. Partial-season outcomes are never targets.
            if len(current_pl) != 380 or len(prior_pl) != 380 or len(prior_ch) != 552:
                continue
            inputs, outputs = strengths(prior_ch), strengths(current_pl)
            for team in sorted((clubs(current_pl) - clubs(prior_pl)) & set(inputs)):
                self.samples.append({'season': season, 'team': team,
                                     'x': inputs[team], 'y': outputs[team]})
        self.model = None
        if len(self.samples) >= 3:
            self.model = Ridge(alpha=self.alpha).fit(
                [r['x'] for r in self.samples], [r['y'] for r in self.samples])
        previous = season_before(target_season)
        self.previous_pl = clubs(pl[pl.season == previous])
        self.ch_strengths = strengths(championship[championship.season == previous])
        self.promoted = set(self.ch_strengths) - self.previous_pl
        if target_teams is not None:
            self.promoted &= set(target_teams)
        return self

    def predict(self, team):
        if team not in self.ch_strengths or self.model is None:
            return (0.0, 0.0)
        return tuple(float(x) for x in self.model.predict([self.ch_strengths[team]])[0])

    def diagnostics(self):
        """Later-season conversion errors using earlier transition cohorts only."""
        rows = []
        for season in sorted({r['season'] for r in self.samples}):
            train = [r for r in self.samples if r['season'] < season]
            test = [r for r in self.samples if r['season'] == season]
            if len(train) < 3:
                continue
            model = Ridge(alpha=self.alpha).fit([r['x'] for r in train], [r['y'] for r in train])
            for row in test:
                prediction = model.predict([row['x']])[0]
                rows.append({'season': season, 'team': row['team'], 'training_teams': len(train),
                             'attack_error': float(prediction[0] - row['y'][0]),
                             'defence_error': float(prediction[1] - row['y'][1])})
        return rows
