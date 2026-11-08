import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import poisson

from epl_predictor.boosting import train_booster, predict_booster, load_booster
from epl_predictor.data import validate, utc, team_name
from epl_predictor.evaluation import metrics, ensemble_weight, temperature
from epl_predictor.features import FeatureBuilder, feature_columns
from epl_predictor.goals import DixonColes, tau, score_matrix, outcomes, reconcile, markets
from epl_predictor.players import ea_units, player_form, fotmob_units, map_player_ids, validate_ea
from epl_predictor.predictor import Predictor
from epl_predictor.promotion import PromotionPrior


def sample_matches():
    rng = np.random.default_rng(4)
    records = []
    for season,year in [('2023-24',2023),('2024-25',2024),('2025-26',2025)]:
        for i,(home,away) in enumerate((h,a) for h in 'ABCD' for a in 'ABCD' if h!=a):
            records.append({'date':f'{year}-08-{i//2+1:02d}','season':season,'home_team':home,'away_team':away,
                            'home_goals':int(rng.poisson(1.6)),'away_goals':int(rng.poisson(1.1)),'status':'FT'})
    return validate(pd.DataFrame(records))


class GoalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = sample_matches()
        cls.model = DixonColes().fit(cls.data,'2025-09-01',extra_teams=['Promoted'])

    def test_tau_known_values(self):
        h,a = np.array([0,0,1,1,2]),np.array([0,1,0,1,2])
        np.testing.assert_allclose(tau(h,a,1.5,1.,-.1),[1.15,.85,.9,1.1,1.])

    def test_independent_poisson_and_normalization(self):
        m = score_matrix(1.5,1.1)
        self.assertAlmostEqual(m[2,1],poisson.pmf(2,1.5)*poisson.pmf(1,1.1),places=10)
        self.assertAlmostEqual(m.sum(),1)
        self.assertAlmostEqual(outcomes(m).sum(),1)
        self.assertAlmostEqual(outcomes(score_matrix(1.2,1.2))[0],outcomes(score_matrix(1.2,1.2))[2])

    def test_dc_preserves_marginals(self):
        corrected = score_matrix(1.5,1.1,-.1)
        independent = score_matrix(1.5,1.1)
        np.testing.assert_allclose(corrected.sum(axis=0),independent.sum(axis=0),atol=1e-11)
        np.testing.assert_allclose(corrected.sum(axis=1),independent.sum(axis=1),atol=1e-11)
        self.assertGreater(corrected[0,0],independent[0,0])
        with self.assertRaises(ValueError): score_matrix(3,3,1)

    def test_market_aggregation_and_ranking(self):
        matrix = score_matrix(2.2,1.3,-.05)
        result = markets(matrix)
        self.assertEqual(result['most_likely_score'],'2-1')
        self.assertEqual(len(result['top_five_scorelines']),5)
        self.assertAlmostEqual(result['over_2_5']+result['under_2_5'],1)
        self.assertAlmostEqual(result['expected_home_goals'],2.2,places=9)
        self.assertAlmostEqual(result['both_teams_to_score'],matrix[1:,1:].sum())
        self.assertTrue(all(result['top_five_scorelines'][i]['probability']>=result['top_five_scorelines'][i+1]['probability'] for i in range(4)))

    def test_reconciliation_is_consistent(self):
        matrix = score_matrix(1.5,1.1,-.1)
        changed = reconcile(matrix,[.6,.25,.15])
        np.testing.assert_allclose(outcomes(changed),[.6,.25,.15])
        self.assertAlmostEqual(changed[2,0]/changed[3,0],matrix[2,0]/matrix[3,0])

    def test_all_pairings_positive_including_unknown(self):
        for home in ['A','B','C','D','Promoted','Unknown']:
            for away in ['A','B','C','D','Promoted','Unknown']:
                if home == away: continue
                lam,mu = self.model.rates(home,away)
                self.assertGreater(min(lam,mu),0)
                self.assertTrue((self.model.matrix(home,away)>=0).all())
        self.assertAlmostEqual(self.model.attack.sum(),0,places=8)
        self.assertAlmostEqual(self.model.defence.sum(),0,places=8)

    def test_reproducible_fit_and_save_load(self):
        other = DixonColes().fit(self.data,'2025-09-01',extra_teams=['Promoted'])
        np.testing.assert_allclose(self.model.attack,other.attack)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.json'
            self.model.save(path)
            loaded = DixonColes.load(path)
            np.testing.assert_allclose(loaded.matrix('A','B'),self.model.matrix('A','B'))

    def test_fit_ignores_future_outcomes(self):
        changed = self.data.copy()
        changed.loc[changed.day >= utc('2025-08-01'),'home_goals'] = 15
        first = DixonColes().fit(self.data,'2025-08-01')
        second = DixonColes().fit(changed,'2025-08-01')
        np.testing.assert_allclose(first.matrix('A','B'),second.matrix('A','B'))


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.data = sample_matches()
        self.ch = self.data.iloc[:0].copy()

    def test_same_day_and_future_mutation_do_not_leak(self):
        first = FeatureBuilder(self.data,self.ch).build()
        changed = self.data.copy()
        cutoff = utc('2025-08-03')
        changed.loc[changed.day >= cutoff,'home_goals'] = 9
        second = FeatureBuilder(changed,self.ch).build()
        cols = feature_columns(first)
        pd.testing.assert_frame_equal(first.loc[first.day <= cutoff,cols],second.loc[second.day <= cutoff,cols])

    def test_features_missing_xg_and_rest(self):
        frame = FeatureBuilder(self.data,self.ch).build()
        self.assertTrue(frame.home_xgf_5.isna().all())
        self.assertTrue(frame.home_xg_count_5.eq(0).all())
        self.assertEqual(frame.iloc[0].home_sample,0)
        self.assertNotIn('target',feature_columns(frame))
        self.assertNotIn('home_goals',feature_columns(frame))
        self.assertTrue(frame.home_rest_days.dropna().ge(1).all())

    def test_team_aliases(self):
        for name in ['Nottm Forest',"Nott'm Forest",'Nottingham Forest']:
            self.assertEqual(team_name(name),'Nottingham')
        self.assertEqual(team_name('Man Utd'),'Man United')
        self.assertEqual(team_name('Coventry City'),'Coventry')
        self.assertEqual(team_name('Totally New Club'),'Totally New Club')

    def test_promotion_insufficient_samples_is_explicit_neutral(self):
        ch = self.data.copy()
        ch['home_team'] = ch.home_team.map(lambda x:'CH'+x)
        ch['away_team'] = ch.away_team.map(lambda x:'CH'+x)
        prior = PromotionPrior().fit(self.data,ch,'2026-27',['CHA'])
        self.assertEqual(prior.promoted,{'CHA'})
        self.assertEqual(prior.predict('CHA'),(0.,0.))
        self.assertEqual(len(prior.samples),0)


class PlayerTests(unittest.TestCase):
    def setUp(self):
        self.ea = pd.DataFrame([
            {'player_id':1,'club':'Arsenal','position':'ST','overall_rating':90,'snapshot_date':'2026-09-12','shooting':95,'attacking_finishing':93},
            {'player_id':2,'club':'Arsenal','position':'CB','overall_rating':70,'snapshot_date':'2026-09-12','defending_awareness':65},
            {'player_id':3,'club':'Arsenal','position':'GK','overall_rating':80,'snapshot_date':'2026-09-12','goalkeeping_reflexes':85},
        ])
        self.fm = pd.DataFrame([
            {'player_id':'a','match_id':1,'date':'2026-08-01','available_at':'2026-08-02','team':'Arsenal','position':'ST','rating':7.,'minutes':90},
            {'player_id':'b','match_id':1,'date':'2026-08-01','available_at':'2026-08-02','team':'Arsenal','position':'ST','rating':10.,'minutes':2},
            {'player_id':'a','match_id':2,'date':'2026-08-03','available_at':'2026-08-04','team':'Arsenal','position':'ST','rating':8.,'minutes':90},
        ])

    def test_snapshot_availability_and_position_aggregation(self):
        self.assertEqual(ea_units(self.ea,'2026-09-12'),{})
        result = ea_units(self.ea,'2026-09-13')['Arsenal']
        self.assertEqual(result['ea_ATT_talent'],94)
        self.assertEqual(result['ea_DEF_talent'],65)
        self.assertIsNone(result['ea_MID_talent'])
        self.assertTrue(result['lineups_estimated'])

    def test_future_snapshot_transfer_and_duplicate_ids(self):
        later = self.ea.iloc[:1].copy()
        later['club'],later['snapshot_date'] = 'Chelsea','2026-10-01'
        combined = pd.concat([self.ea,later])
        self.assertIn('Arsenal',ea_units(combined,'2026-09-14'))
        self.assertEqual(ea_units(combined,'2026-10-02')['Chelsea']['ea_ATT_players'],1)
        with self.assertRaises(ValueError): validate_ea(pd.concat([self.ea,self.ea]))

    def test_minutes_weighting_shrinkage_and_future_cutoff(self):
        forms = player_form(self.fm,'2026-08-03').set_index('player_id')
        self.assertEqual(forms.loc['a','appearances'],1)
        self.assertLess(forms.loc['b','smoothed_rating'],8)
        after = player_form(self.fm,'2026-08-05').set_index('player_id')
        self.assertEqual(after.loc['a','rating_5'],7.5)
        self.assertEqual(after.loc['a','appearances'],2)
        self.assertIn('fm_ATT_form',fotmob_units(self.fm,'2026-08-05')['Arsenal'])

    def test_missing_ratings_and_identical_names_not_invented(self):
        missing = self.fm.copy()
        missing['rating'] = np.nan
        self.assertTrue(player_form(missing,'2026-08-05').empty)
        bridge = pd.DataFrame({'ea_id':['1'],'fotmob_id':['a']})
        accepted,unmatched = map_player_ids(self.ea,self.fm,bridge)
        self.assertEqual(accepted,{'a':'1'})
        self.assertEqual(unmatched,[])
        with self.assertRaises(ValueError):
            map_player_ids(self.ea,self.fm,pd.concat([bridge,bridge]))


class ModelIntegrationTests(unittest.TestCase):
    def test_xgboost_native_persistence_missing_features(self):
        rng = np.random.default_rng(2)
        frame = pd.DataFrame({'a':rng.normal(size=30),'b':rng.normal(size=30),'target':np.tile([0,1,2],10)})
        frame.loc[0,'b'] = np.nan
        model = train_booster(frame,['a','b'],{'n_estimators':5,'max_depth':2})
        p = predict_booster(model,frame,['a','b'])
        np.testing.assert_allclose(p.sum(axis=1),1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.json'
            model.save_model(path)
            np.testing.assert_allclose(p,predict_booster(load_booster(path),frame,['a','b']))
        with self.assertRaises(ValueError):predict_booster(model,frame.drop(columns='b'),['a','b'])

    def test_ensemble_and_calibration(self):
        y = np.array([0,1,2]*5)
        good = np.eye(3)[y]*.7+.1
        poor = np.ones((15,3))/3
        weight,_ = ensemble_weight(y,poor,good)
        self.assertEqual(weight,1)
        scaled = temperature(good,1.5)
        np.testing.assert_allclose(scaled.sum(axis=1),1)
        self.assertLess(metrics(y,good)['log_loss'],metrics(y,poor)['log_loss'])

    def test_saved_predictor_end_to_end_and_missing_booster_fallback(self):
        pl = sample_matches()
        ch = pl.iloc[:0].copy()
        model = DixonColes().fit(pl,'2025-09-01')
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            model.save(folder/'dixon_coles.json')
            pl.to_csv(folder/'pl_history.csv',index=False)
            ch.to_csv(folder/'championship_history.csv',index=False)
            manifest = {'schema_version':1,'cutoff':'2025-09-01','trained_at':'2025-09-01',
                        'selected_model':'xgb_base','teams':list('ABCD'),'feature_schemas':{'xgb_base':['missing']},
                        'player_model_status':'unavailable','selection_metrics':{},'test_metrics':{}}
            (folder/'manifest.json').write_text(json.dumps(manifest))
            predictor = Predictor(folder)
            fixtures = pd.DataFrame([{'home_team':'A','away_team':'B','kickoff_utc':'2025-09-10'}])
            result = predictor.predict(fixtures)[0]
            self.assertEqual(result['model'],'dixon_coles_fallback')
            self.assertEqual(len(result['top_five_scorelines']),5)
            self.assertIsNone(result['player_context']['A']['ea'])
            self.assertAlmostEqual(result['home_win_probability']+result['draw_probability']+result['away_win_probability'],1)
            fixtures['kickoff_utc'] = '2025-08-01'
            with self.assertRaises(ValueError):predictor.predict(fixtures)


if __name__ == '__main__':
    unittest.main()
