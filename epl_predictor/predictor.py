"""Load saved models and produce internally consistent, explained match forecasts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost.core import XGBoostError

from .boosting import load_booster, predict_booster
from .data import team_name, utc, validate
from .evaluation import temperature
from .features import FeatureBuilder
from .goals import DixonColes, markets, outcomes, reconcile, score_matrix
from .players import ea_units, fotmob_units


class Predictor:
    def __init__(self, artifact_dir='artifacts/hybrid_v1'):
        self.folder = Path(artifact_dir)
        required_files = ['manifest.json', 'dixon_coles.json', 'pl_history.csv', 'championship_history.csv']
        missing = [f for f in required_files if not (self.folder / f).exists()]
        if missing:
            raise FileNotFoundError(f'Missing required artifact files in {artifact_dir}: {missing}')
        self.manifest = json.loads((self.folder/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('schema_version') != 1:
            raise ValueError('Unsupported artifact schema')
        self.cutoff = utc(self.manifest['cutoff'])
        self.dc = DixonColes.load(self.folder/'dixon_coles.json')
        self.pl = validate(pd.read_csv(self.folder/'pl_history.csv'))
        self.ch = validate(pd.read_csv(self.folder/'championship_history.csv'))
        if (self.pl.available_at > self.cutoff).any() or (self.ch.available_at > self.cutoff).any():
            raise ValueError('Artifact contains results after its cutoff')
        self.boosters = {}

    def predict(self, fixtures, ea=None, appearances=None, filter_future=False):
        fixtures = fixtures.copy()
        for col in ['home_team','away_team']:
            fixtures[col] = fixtures[col].map(team_name)
        fixtures['kickoff_utc'] = pd.to_datetime(fixtures.kickoff_utc,utc=True,format='mixed')
        if filter_future:
            fixtures = fixtures[fixtures.kickoff_utc >= self.cutoff].reset_index(drop=True)
            if fixtures.empty:
                return []
        elif (fixtures.kickoff_utc < self.cutoff).any():
            raise ValueError('Cannot predict before the saved model cutoff; use chronological evaluation')
        if fixtures.home_team.eq(fixtures.away_team).any():
            raise ValueError('Home and away clubs must differ')
        if 'season' not in fixtures:
            fixtures['season'] = fixtures.kickoff_utc.map(lambda d:f'{d.year if d.month >= 7 else d.year-1}-{str(d.year+1 if d.month >= 7 else d.year)[-2:]}')
        trained_season = self.pl.season.max()
        if not fixtures.season.eq(trained_season).all():
            raise ValueError(f'Saved model covers {trained_season}; retrain for a new season')
        frame = FeatureBuilder(self.pl,self.ch).upcoming(fixtures,self.cutoff)
        selected = self.manifest['selected_model']
        name = selected.removesuffix('_calibrated')
        boost_p = None
        fallback = None
        if name.startswith(('xgb_','hybrid_')):
            booster_name = name.replace('hybrid_','xgb_',1)
            try:
                if booster_name not in self.boosters:
                    self.boosters[booster_name] = load_booster(self.folder/f'{booster_name}.json')
                columns = self.manifest['feature_schemas'][booster_name]
                boost_p = predict_booster(self.boosters[booster_name],frame,columns)
            except (OSError,ValueError,KeyError,XGBoostError) as error:
                fallback = f'Using Dixon-Coles because outcome features/model are unavailable: {error}'
        talent = ea_units(ea,self.cutoff) if ea is not None else {}
        forms = fotmob_units(appearances,self.cutoff) if appearances is not None else {}
        baseline = None
        if name == 'poisson':
            from poisson_distribution import build_predictions
            season = self.pl.season.max()
            baseline,_ = build_predictions(self.pl[self.pl.season < season],self.ch[self.ch.season < season],
                                            self.pl[self.pl.season == season],fixtures)
        results = []
        for i,row in enumerate(fixtures.itertuples()):
            known = row.home_team in self.manifest['teams'] and row.away_team in self.manifest['teams']
            reason = fallback if known else 'Club outside current roster: goal-model fallback; unseen identities use neutral strength'
            active = selected if reason is None else 'dixon_coles_fallback'
            matrix = self.dc.matrix(row.home_team,row.away_team)
            p = outcomes(matrix)
            if reason is None:
                if name == 'poisson':
                    b = baseline.iloc[i]
                    matrix = score_matrix(b.expected_home_goals,b.expected_away_goals)
                    p = outcomes(matrix)
                elif name.startswith('xgb_'):
                    p = boost_p[i]
                elif name.startswith('hybrid_'):
                    weight = self.manifest['ensemble_weights'][name]
                    p = (1-weight)*p+weight*boost_p[i]
                if selected.endswith('_calibrated'):
                    p = temperature(p[None,:],self.manifest['temperatures'][name])[0]
            coherent = reconcile(matrix,p)
            statistics = markets(coherent)
            components = frame.iloc[i]
            results.append({'home_team':row.home_team,'away_team':row.away_team,
                'kickoff_utc':row.kickoff_utc.isoformat(),**statistics,
                'probability_units':'fraction (multiply by 100 for percent)',
                'model':active,'data_cutoff':self.manifest['cutoff'],'trained_at':self.manifest['trained_at'],
                'fallback_reason':reason,'promotion_prior_teams':[t for t in [row.home_team,row.away_team] if t in self.dc.promoted],
                'score_distribution':'H/D/A-region reconciliation of goal model; all markets use the same matrix',
                'player_features_used_in_model':False,'player_model_status':self.manifest['player_model_status'],
                'player_context':{t:{'ea':talent.get(t),'fotmob':forms.get(t)} for t in [row.home_team,row.away_team]},
                'explanation_inputs':{'goal_model_home_log_advantage':self.dc.home_advantage,
                                      'pre_match_elo_difference':float(components.diff_elo),
                                      'home_prior_league_matches':int(components.home_sample),
                                      'away_prior_league_matches':int(components.away_sample),
                                      'interpretation':'Observed inputs and fitted coefficients; not a causal attribution'},
                'evaluation':{'selection':self.manifest['selection_metrics'],'test':self.manifest['test_metrics']}})
        return results
