"""Tune, calibrate, select and test on disjoint chronological periods."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .boosting import train_booster, predict_booster
from .data import clubs, inventory, load_data, utc
from .evaluation import metrics, reliability, temperature, fit_temperature, ensemble_weight
from .features import FeatureBuilder, feature_columns
from .goals import DixonColes, score_matrix, outcomes, reconcile, markets
from .promotion import PromotionPrior

GROUPS = {'base':['base'], 'base_rest':['base','rest'],
          'base_promotion':['base','promotion'], 'base_xg':['base','xg'],
          'full':['base','rest','promotion','xg','h2h']}


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def phase(day, config):
    if day < utc(config['ensemble_start']): return 'tune'
    if day < utc(config['calibration_start']): return 'ensemble'
    if day < utc(config['selection_start']): return 'calibration'
    if day < utc(config['test_start']): return 'selection'
    if day < utc(config['current_start']): return 'test'
    return 'current_replay'


def baseline_matrices(pl, ch, block, origin):
    """Preserve and evaluate the original project Poisson as a separate baseline."""
    from poisson_distribution import build_predictions
    season = block.season.iloc[0]
    historical = pl[pl.season < season]
    current = pl[(pl.season == season) & (pl.available_at <= origin)]
    fixtures = block[['day','season','home_team','away_team']].rename(columns={'day':'kickoff_utc'})
    prediction, _ = build_predictions(historical,ch[ch.season < season],current,fixtures)
    return [score_matrix(row.expected_home_goals,row.expected_away_goals,min_goals=int(max(actual.home_goals,actual.away_goals)))
            for row,actual in zip(prediction.itertuples(),block.itertuples())]


def replay_goal_model(pl,ch,features,settings,report_progress=True):
    matrices = {}
    baseline = {}
    for key, block in features.groupby(features.day.dt.strftime('%Y-%m'),sort=True):
        origin = utc(key+'-01')
        season = block.season.iloc[0]
        roster = clubs(pl[pl.season == season])
        prior = PromotionPrior().fit(pl[pl.available_at <= origin],ch[ch.available_at <= origin],season,roster)
        model = DixonColes(**settings).fit(pl,origin,prior,roster)
        for row in block.itertuples():
            matrices[row.row_id] = model.matrix(row.home_team,row.away_team,int(max(row.home_goals,row.away_goals)))
        baseline.update(zip(block.row_id,baseline_matrices(pl,ch,block,origin)))
        if report_progress:
            print(f'Goal replay {key}: {len(block)} matches',flush=True)
    return matrices, baseline


def train(root=Path('.'), config_path=Path('hybrid_config.json'), artifacts=Path('artifacts/hybrid_v1'), reports=Path('reports')):
    root, artifacts, reports = Path(root),Path(artifacts),Path(reports)
    if artifacts.exists() and (artifacts/'manifest.json').exists():
        raise FileExistsError(f'{artifacts} already has a model. Choose a new --artifacts version.')
    artifacts.mkdir(parents=True,exist_ok=True)
    reports.mkdir(parents=True,exist_ok=True)
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    inventory(root,reports/'data_inventory.json')
    pl,ch = load_data(root,config['as_of'])
    print('Generating chronological features...',flush=True)
    features = FeatureBuilder(pl,ch).build()
    features.to_csv(reports/'prematch_features.csv',index=False)
    tune = features[features.season == config['tuning_season']]
    tune_train = features[(features.season < config['tuning_season']) & (features.season >= '2020-21')]
    if tune.empty or tune_train.empty:
        raise ValueError('Configured tuning period has no data')
    tuning = []
    selected_dc, best_loss = None,float('inf')
    for settings in config['dc_candidates']:
        print(f'Tuning Dixon-Coles {settings}',flush=True)
        matrices,_ = replay_goal_model(pl,ch,tune,settings,False)
        score = metrics(tune.target,np.array([outcomes(matrices[r]) for r in tune.row_id]))
        tuning.append({'kind':'dixon_coles','settings':settings,**score})
        if score['log_loss'] < best_loss:
            selected_dc,best_loss = settings,score['log_loss']
    selected_xgb,best_loss = None,float('inf')
    all_columns = feature_columns(features)
    for settings in config['xgb_candidates']:
        model = train_booster(tune_train,all_columns,settings,config['seed'])
        score = metrics(tune.target,predict_booster(model,tune,all_columns))
        tuning.append({'kind':'xgboost','settings':settings,**score})
        if score['log_loss'] < best_loss:
            selected_xgb,best_loss = settings,score['log_loss']
    write_json(reports/'tuning.json',tuning)
    print(f'Selected DC {selected_dc}; XGBoost {selected_xgb}',flush=True)
    evaluation = features[features.day >= utc(config['ensemble_start'])].copy()
    matrices, baseline = replay_goal_model(pl,ch,evaluation,selected_dc)
    dc = np.array([outcomes(matrices[r]) for r in evaluation.row_id])
    poisson = np.array([outcomes(baseline[r]) for r in evaluation.row_id])
    predictions = {'poisson':poisson,'dixon_coles':dc}
    # Monthly expanding-window refits; all candidates see identical test fixtures.
    for name,groups in GROUPS.items():
        cols = feature_columns(features,groups)
        blocks = []
        for key,block in evaluation.groupby(evaluation.day.dt.strftime('%Y-%m'),sort=True):
            origin = utc(key+'-01')
            training = features[(features.day < origin) & (features.season >= '2020-21')]
            model = train_booster(training,cols,selected_xgb,config['seed'])
            blocks.append(predict_booster(model,block,cols))
        predictions['xgb_'+name] = np.concatenate(blocks)
        print(f'XGBoost replay complete: {name}',flush=True)
    stages = np.array([phase(d,config) for d in evaluation.day])
    y = evaluation.target.to_numpy()
    ensemble_mask,calibration_mask = stages=='ensemble',stages=='calibration'
    weights,weight_search = {},[]
    for name in GROUPS:
        weight,search = ensemble_weight(y[ensemble_mask],dc[ensemble_mask],predictions['xgb_'+name][ensemble_mask])
        weights['hybrid_'+name] = weight
        predictions['hybrid_'+name] = (1-weight)*dc+weight*predictions['xgb_'+name]
        weight_search.extend([{'variant':name,**r} for r in search])
    write_json(reports/'ensemble_search.json',weight_search)
    temperatures = {}
    for name,p in list(predictions.items()):
        t = fit_temperature(y[calibration_mask],p[calibration_mask])
        temperatures[name] = t
        predictions[name+'_calibrated'] = temperature(p,t)
    selection = stages=='selection'
    selection_scores = {name:metrics(y[selection],p[selection]) for name,p in predictions.items()}
    # Selection is frozen here, before evaluating test-period metrics.
    selected = min(selection_scores,key=lambda name:(selection_scores[name]['log_loss'],selection_scores[name]['brier']))
    print(f'Frozen selection from 2024-25: {selected}',flush=True)
    comparisons, calibration, match_rows = [],[],[]
    for name,p in predictions.items():
        for stage in ['ensemble','calibration','selection','test','current_replay']:
            mask = stages==stage
            if not mask.any(): continue
            score = metrics(y[mask],p[mask])
            nll = []
            exact = []
            for i in np.flatnonzero(mask):
                row = evaluation.iloc[i]
                base = baseline[row.row_id] if name.startswith('poisson') else matrices[row.row_id]
                coherent = reconcile(base,p[i])
                nll.append(-np.log(max(coherent[int(row.home_goals),int(row.away_goals)],1e-15)))
                exact.append(np.unravel_index(coherent.argmax(),coherent.shape)==(row.home_goals,row.away_goals))
            comparisons.append({'model':name,'phase':stage,**score,'score_log_loss':float(np.mean(nll)),
                                'exact_score_accuracy':float(np.mean(exact)),'selected':name==selected})
            calibration.extend([{'model':name,'phase':stage,**r} for r in reliability(y[mask],p[mask])])
        for i,row in enumerate(evaluation.itertuples()):
            match_rows.append({'row_id':row.row_id,'day':row.day,'model':name,'phase':stages[i],
                               'actual':int(row.target),'home_probability':p[i,0],'draw_probability':p[i,1],
                               'away_probability':p[i,2]})
    pd.DataFrame(comparisons).to_csv(reports/'model_comparison.csv',index=False)
    pd.DataFrame(calibration).to_csv(reports/'calibration.csv',index=False)
    pd.DataFrame(match_rows).to_csv(reports/'evaluation_predictions.csv',index=False)
    # Outcome subgroups use predictions only to define favorites, and real membership for promotion.
    subgroups = []
    for name,p in predictions.items():
        for stage in ['selection','test','current_replay']:
            mask = stages==stage
            promoted = evaluation.home_promoted.to_numpy().astype(bool) | evaluation.away_promoted.to_numpy().astype(bool)
            for label,group in [('promoted',promoted),('established',~promoted),
                                ('strong_favorite',p.max(axis=1)>=.6),('competitive',p.max(axis=1)<.6),
                                ('underdog_won',p[np.arange(len(y)),y]<.25)]:
                chosen = mask & group
                if chosen.any(): subgroups.append({'model':name,'phase':stage,'subgroup':label,**metrics(y[chosen],p[chosen])})
    pd.DataFrame(subgroups).to_csv(reports/'subgroups.csv',index=False)
    pd.DataFrame([r for r in comparisons if r['model'].startswith('xgb_')]).to_csv(reports/'feature_ablations.csv',index=False)
    # Permute features only on selection-period predictions from a pre-period model.
    train_imp = features[(features.day < utc(config['selection_start'])) & (features.season >= '2020-21')]
    select_imp = evaluation[selection]
    imp_model = train_booster(train_imp,all_columns,selected_xgb,config['seed'])
    base_loss = metrics(select_imp.target,predict_booster(imp_model,select_imp,all_columns))['log_loss']
    rng = np.random.default_rng(config['seed'])
    importance = []
    for col in all_columns:
        changed = select_imp.copy()
        changed[col] = rng.permutation(changed[col].to_numpy())
        loss = metrics(changed.target,predict_booster(imp_model,changed,all_columns))['log_loss']
        importance.append({'feature':col,'permuted_log_loss_increase':loss-base_loss})
    pd.DataFrame(importance).sort_values('permuted_log_loss_increase',ascending=False).to_csv(reports/'permutation_importance.csv',index=False)
    cutoff = utc(config['as_of']).normalize()
    season = pl.season.max()
    roster = clubs(pl[pl.season == season])
    prior = PromotionPrior().fit(pl,ch,season,roster)
    model = DixonColes(**selected_dc).fit(pl,cutoff,prior,roster)
    model.save(artifacts/'dixon_coles.json')
    write_json(reports/'promotion_validation.json',{'samples':prior.samples,'chronological_errors':prior.diagnostics()})
    for name,groups in GROUPS.items():
        columns = feature_columns(features,groups)
        fitted = train_booster(features[features.season >= '2020-21'],columns,selected_xgb,config['seed'])
        fitted.save_model(artifacts/f'xgb_{name}.json')
    # Snapshot the exact match inputs so subsequent raw refreshes cannot silently change inference.
    pl.to_csv(artifacts/'pl_history.csv',index=False)
    ch.to_csv(artifacts/'championship_history.csv',index=False)
    manifest = {'schema_version':1,'selected_model':selected,'cutoff':cutoff.isoformat(),
                'trained_at':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
                'config':config,'dc_settings':selected_dc,'xgb_settings':selected_xgb,
                'feature_schemas':{'xgb_'+k:feature_columns(features,v) for k,v in GROUPS.items()},
                'ensemble_weights':weights,'temperatures':temperatures,'teams':sorted(roster),
                'selection_metrics':selection_scores[selected],
                'test_metrics':next(r for r in comparisons if r['model']==selected and r['phase']=='test'),
                'current_replay_metrics':next((r for r in comparisons if r['model']==selected and r['phase']=='current_replay'),None),
                'fallback_model':'dixon_coles','fallback_reason':'Missing feature/schema or unknown club',
                'player_model_status':'experimental: one EA snapshot; no dated FotMob appearances; no independent player training/validation split',
                'unavailable_ablations':['EA talent','FotMob player form','position-aware player units','EA + FotMob'],
                'input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'premier_league_match_history.csv',root/'championship_match_history.csv',root/'current_premier_league_results.csv',root/'players.csv'] if p.exists()}}
    write_json(artifacts/'manifest.json',manifest)
    write_json(reports/'run_manifest.json',manifest)
    print(pd.DataFrame(comparisons).query("phase == 'test'")[['model','matches','log_loss','brier','accuracy']].to_string(index=False),flush=True)
    return manifest
