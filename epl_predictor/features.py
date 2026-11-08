"""Daily, chronological pre-match features. Same-day outcomes are never visible."""
from collections import defaultdict

import numpy as np
import pandas as pd

from .data import clubs, utc
from .promotion import PromotionPrior


def mean(values):
    return float(np.mean(values)) if len(values) else np.nan


class FeatureBuilder:
    def __init__(self, matches, championship):
        self.matches, self.championship = matches, championship
        self.history = defaultdict(list)
        self.elo = defaultdict(lambda: 1500.)
        self.h2h = defaultdict(list)
        self.league_goals = []
        self.prior_cache = {}

    def _prior(self, season):
        if season not in self.prior_cache:
            # Team membership is schedule information; no target-season scores enter fitting.
            teams = clubs(self.matches[self.matches.season == season])
            prior = PromotionPrior().fit(self.matches, self.championship, season, teams or None)
            self.prior_cache[season] = prior
        return self.prior_cache[season]

    def _team(self, team, day, season, home):
        prior = self._prior(season)
        promoted = team in prior.promoted
        history = self.history[team]
        if promoted:
            history = [r for r in history if r['season'] == season]
        out = {'elo': self.elo[team], 'sample': len(history)}
        for size in [5, 10]:
            rows = history[-size:]
            for column in ['gf','ga','points','adjusted_gd','xgf','xga']:
                values = [r[column] for r in rows if np.isfinite(r[column])]
                out[f'{column}_{size}'] = mean(values)
            out[f'xg_count_{size}'] = sum(np.isfinite(r['xgf']) for r in rows)
            out[f'gd_{size}'] = out[f'gf_{size}'] - out[f'ga_{size}']
        recent = history[-10:]
        if recent:
            w = np.exp(-np.log(2) * np.array([(day-r['day']).days for r in recent])/60)
            out['weighted_points'] = float(np.average([r['points'] for r in recent], weights=w))
        else:
            out['weighted_points'] = np.nan
        venue = [r for r in history if r['home'] == home][-10:]
        out['venue_gf'], out['venue_ga'] = mean([r['gf'] for r in venue]), mean([r['ga'] for r in venue])
        out['rest_days'] = float((day-history[-1]['day']).days) if history else np.nan
        for days in [7,14]:
            out[f'games_{days}'] = sum(0 < (day-r['day']).days <= days for r in history)
        pa, pd_ = prior.predict(team) if promoted else (0.,0.)
        out.update(promoted=float(promoted), promotion_attack=pa, promotion_defence=pd_,
                   promotion_samples=len(prior.samples))
        return out

    def fixture(self, home, away, day, season):
        home_stats, away_stats = self._team(home, day, season, True), self._team(away, day, season, False)
        result = {}
        for key in home_stats:
            result[f'home_{key}'], result[f'away_{key}'] = home_stats[key], away_stats[key]
            result[f'diff_{key}'] = home_stats[key] - away_stats[key]
        last = self.league_goals[-380:]
        result['league_home_goals'] = mean([r[0] for r in last])
        result['league_away_goals'] = mean([r[1] for r in last])
        encounters = self.h2h[tuple(sorted([home,away]))][-5:]
        if encounters:
            w = np.exp(-np.log(2)*np.array([(day-r[0]).days for r in encounters])/365)
            result['h2h_goal_difference'] = float(np.average([r[2] if r[1] == home else -r[2] for r in encounters],weights=w))
        else:
            result['h2h_goal_difference'] = np.nan
        result['h2h_samples'] = len(encounters)
        return result

    def update_day(self, frame):
        # Elo updates also use pre-day ratings, avoiding ordering effects for simultaneous games.
        changes = defaultdict(float)
        for row in frame.itertuples():
            expected = 1 / (1 + 10 ** ((self.elo[row.away_team] - self.elo[row.home_team])/400))
            result = 1. if row.home_goals > row.away_goals else .5 if row.home_goals == row.away_goals else 0.
            change = 20 * (result - expected)
            changes[row.home_team] += change
            changes[row.away_team] -= change
            for team, other, gf, ga, xgf, xga, home in [
                (row.home_team,row.away_team,row.home_goals,row.away_goals,getattr(row,'home_xg',np.nan),getattr(row,'away_xg',np.nan),True),
                (row.away_team,row.home_team,row.away_goals,row.home_goals,getattr(row,'away_xg',np.nan),getattr(row,'home_xg',np.nan),False)]:
                self.history[team].append({'day':row.day,'season':row.season,'gf':gf,'ga':ga,
                    'points':3 if gf > ga else 1 if gf == ga else 0, 'home':home,'xgf':xgf,'xga':xga,
                    'adjusted_gd':gf-ga+(self.elo[other]-1500)/400})
            self.h2h[tuple(sorted([row.home_team,row.away_team]))].append((row.day,row.home_team,row.home_goals-row.away_goals))
            self.league_goals.append((row.home_goals,row.away_goals))
        for team, change in changes.items():
            self.elo[team] += change

    def build(self):
        records = []
        for day, group in self.matches.groupby('day', sort=True):
            for row in group.itertuples():
                features = self.fixture(row.home_team,row.away_team,day,row.season)
                features.update(row_id=f'{row.season}:{row.home_team}:{row.away_team}',
                                day=day, season=row.season, home_team=row.home_team, away_team=row.away_team,
                                target=row.target, home_goals=row.home_goals, away_goals=row.away_goals)
                records.append(features)
            self.update_day(group)
        return pd.DataFrame(records)

    def upcoming(self, fixtures, cutoff):
        cutoff = utc(cutoff).normalize()
        for day, group in self.matches[self.matches.available_at <= cutoff].groupby('day',sort=True):
            self.update_day(group)
        records = []
        for row in fixtures.itertuples():
            day = utc(row.kickoff_utc).normalize()
            if day < cutoff:
                raise ValueError('Prediction date precedes the artifact information cutoff; use backtest')
            records.append(self.fixture(row.home_team,row.away_team,day,row.season))
        return pd.DataFrame(records)


def feature_columns(frame, groups=('base','rest','promotion','xg','h2h')):
    metadata = {'row_id','day','season','home_team','away_team','target','home_goals','away_goals'}
    result = []
    for col in frame.columns:
        if col in metadata:
            continue
        group = ('rest' if any(k in col for k in ['rest_days','games_7','games_14']) else
                 'promotion' if any(k in col for k in ['promoted','promotion_']) else
                 'xg' if 'xg' in col else 'h2h' if col.startswith('h2h') else 'base')
        if group in groups:
            result.append(col)
    return result
