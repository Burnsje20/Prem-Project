"""Optional, explicitly dated player features. Missing ratings are never fabricated."""
import numpy as np
import pandas as pd

from .data import team_name, utc

UNITS = {'GK':'GK','CB':'DEF','LB':'DEF','RB':'DEF','LWB':'DEF','RWB':'DEF',
         'CDM':'MID','CM':'MID','CAM':'MID','LM':'MID','RM':'MID',
         'ST':'ATT','CF':'ATT','LW':'ATT','RW':'ATT'}
ATTRIBUTES = {
    'GK':['goalkeeping_diving','goalkeeping_handling','goalkeeping_reflexes','goalkeeping_positioning'],
    'DEF':['defending_awareness','mentality_interceptions','defending_standing_tackle','defending_sliding_tackle','power_strength'],
    'MID':['attacking_short_passing','skill_long_passing','mentality_vision','skill_ball_control','skill_dribbling'],
    'ATT':['attacking_finishing','mentality_positioning','shooting','skill_dribbling','pace'],
}


def unit(position):
    return UNITS.get(str(position).upper().strip())


def validate_ea(frame):
    d = frame.copy()
    required = {'player_id','club','position','overall_rating','snapshot_date'}
    if not required <= set(d):
        raise ValueError(f'EA missing {required - set(d)}')
    if d[list(required)].isna().any().any():
        raise ValueError('EA identifiers, availability and ratings must be known')
    d['player_id'] = d.player_id.astype(str)
    d['snapshot_date'] = pd.to_datetime(d.snapshot_date, utc=True, errors='raise')
    if d.duplicated(['player_id','snapshot_date']).any():
        raise ValueError('Ambiguous EA player/snapshot identity')
    if 'gender' in d:
        d = d[d.gender.eq("Men's Football")].copy()
    d['club'] = d.club.map(team_name)
    d['unit'] = d.position.map(unit)
    return d


def ea_units(frame, cutoff, expected_minutes=None):
    """Snapshot-safe positional talent; recent minutes weights need an explicit EA-ID map."""
    d = validate_ea(frame)
    d = d[d.snapshot_date + pd.Timedelta(days=1) <= utc(cutoff).normalize()]
    # Resolve transfers by the latest eligible snapshot BEFORE selecting the club.
    d = d.sort_values('snapshot_date').drop_duplicates('player_id', keep='last')
    result = {}
    for club, squad in d.groupby('club'):
        row = {'lineups_estimated': True, 'talent_proxy': 'position_depth' if expected_minutes is None else 'expected_minutes',
               'unknown_positions': int(squad.unit.isna().sum()), 'snapshot_available': True}
        for group in ['GK','DEF','MID','ATT']:
            players = squad[squad.unit.eq(group)].copy()
            if expected_minutes is None:
                # Positional depth proxy, not an assertion about the starting lineup.
                players = players.nlargest(1 if group == 'GK' else 3, 'overall_rating')
                weights = np.ones(len(players))
            else:
                weights = players.player_id.map(expected_minutes).fillna(0).to_numpy(float)
            cols = [c for c in ATTRIBUTES[group] if c in players]
            rating = players[cols].apply(pd.to_numeric, errors='coerce').mean(axis=1) if cols else players.overall_rating
            valid = rating.notna().to_numpy() & (weights > 0)
            row[f'ea_{group}_talent'] = float(np.average(rating[valid],weights=weights[valid])) if valid.any() else None
            row[f'ea_{group}_players'] = int(valid.sum())
            row[f'ea_{group}_overall'] = float(np.average(players.overall_rating[valid],weights=weights[valid])) if valid.any() else None
        result[club] = row
    return result


def map_player_ids(ea, appearances, bridge):
    """Only explicit, one-to-one provider ID links; never fuzzy-match player names."""
    required = {'ea_id','fotmob_id'}
    if not required <= set(bridge):
        raise ValueError('Player bridge requires ea_id and fotmob_id')
    b = bridge.copy().astype(str)
    if b.ea_id.duplicated().any() or b.fotmob_id.duplicated().any():
        raise ValueError('Ambiguous cross-provider player identity')
    ea_ids, fm_ids = set(ea.player_id.astype(str)), set(appearances.player_id.astype(str))
    accepted = b[b.ea_id.isin(ea_ids) & b.fotmob_id.isin(fm_ids)]
    return dict(zip(accepted.fotmob_id,accepted.ea_id)), b.loc[~b.index.isin(accepted.index)].to_dict('records')


def validate_appearances(frame):
    required = {'player_id','match_id','date','available_at','team','position','rating','minutes'}
    if not required <= set(frame):
        raise ValueError(f'Player appearances missing {required - set(frame)}')
    d = frame.copy()
    if d[['player_id','match_id','date','available_at','team','position','minutes']].isna().any().any():
        raise ValueError('Player identity, dates and minutes must be known')
    d['player_id'], d['match_id'] = d.player_id.astype(str), d.match_id.astype(str)
    if d.duplicated(['player_id','match_id']).any():
        raise ValueError('Duplicate player-match record')
    for col in ['date','available_at']:
        d[col] = pd.to_datetime(d[col],utc=True, errors='raise')
    if (d.available_at < d.date).any():
        raise ValueError('Performance cannot be available before the match')
    for col in ['minutes','rating']:
        d[col] = pd.to_numeric(d[col],errors='raise')
    if not d.minutes.between(0,130).all() or not d.rating.dropna().between(0,10).all():
        raise ValueError('Invalid minutes or rating')
    d['team'], d['unit'] = d.team.map(team_name), d.position.map(unit)
    return d.sort_values(['date','match_id','player_id'])


def player_form(frame, cutoff, prior_games=2):
    """Dated ratings with minutes-based shrinkage; experimental prior strength."""
    d = validate_appearances(frame)
    cutoff = utc(cutoff).normalize()
    d = d[(d.date < cutoff) & (d.available_at <= cutoff) & (d.minutes > 0)]
    if d.empty:
        return pd.DataFrame()
    rows = []
    for player, appearances in d.groupby('player_id'):
        valid = appearances.dropna(subset=['rating'])
        if valid.empty:
            continue
        latest = appearances.iloc[-1]
        peers = d[d.unit.eq(latest.unit)].dropna(subset=['rating'])
        prior = float(np.average(peers.rating, weights=peers.minutes))
        recent = valid.tail(10)
        weights = recent.minutes.to_numpy() * np.exp(-np.log(2)*(cutoff-recent.date).dt.total_seconds().to_numpy()/86400/60)
        exposure = float(weights.sum()/90)
        weighted = float(np.average(recent.rating, weights=weights))
        long_term = float(np.average(valid.rating, weights=valid.minutes))
        home_away = {}
        if 'is_home' in valid:
            for label, flag in [('home',True),('away',False)]:
                subset = valid[valid.is_home.eq(flag)].tail(10)
                home_away[label] = float(np.average(subset.rating,weights=subset.minutes)) if len(subset) else np.nan
        rows.append({'player_id':player, 'team':latest.team, 'unit':latest.unit,
                     'rating_5':float(valid.tail(5).rating.mean()),'rating_10':float(recent.rating.mean()),
                     'minutes_weighted_rating':float(np.average(recent.rating,weights=recent.minutes)),
                     'recency_weighted_rating':weighted,
                     'smoothed_rating':(weighted*exposure+prior*prior_games)/(exposure+prior_games),
                     'long_term_rating':long_term,'form_difference':weighted-long_term,
                     'appearances':len(valid),'effective_games':exposure,
                     'expected_minutes':float(appearances.tail(5).minutes.mean()),
                     'home_away_difference':home_away.get('home',np.nan)-home_away.get('away',np.nan)})
    return pd.DataFrame(rows)


def fotmob_units(frame, cutoff):
    forms = player_form(frame, cutoff)
    result = {}
    if forms.empty:
        return result
    for team, players in forms.groupby('team'):
        row = {'lineups_estimated':True, 'rated_players':len(players)}
        for group in ['GK','DEF','MID','ATT']:
            p = players[players.unit.eq(group)]
            row[f'fm_{group}_form'] = float(np.average(p.smoothed_rating,weights=p.expected_minutes)) if len(p) else None
        row['fm_team_form'] = float(np.average(players.smoothed_rating,weights=players.expected_minutes))
        result[team] = row
    return result
