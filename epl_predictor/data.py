"""Validated source loading. Raw sources remain unchanged."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from fotmob_client import normalize_team_name

ALIASES = {"Man Utd": "Man United", "Newcastle Utd": "Newcastle",
           "Nott'm Forest": "Nottingham", "Nottm Forest": "Nottingham",
           "Spurs": "Tottenham", "Leicester City": "Leicester",
           "Norwich City": "Norwich", "Luton Town": "Luton"}


def team_name(value):
    if pd.isna(value) or not str(value).strip():
        raise ValueError("Missing team name")
    name = str(value).strip()
    return normalize_team_name(ALIASES.get(name, name))


def utc(value):
    return pd.to_datetime(value, utc=True)


def season_before(season):
    year = int(season[:4])
    return f"{year - 1}-{str(year)[-2:]}"


def clubs(frame):
    return set(frame.home_team) | set(frame.away_team)


def validate(frame):
    d = frame.copy()
    required = ['date', 'season', 'home_team', 'away_team', 'home_goals', 'away_goals']
    if set(required) - set(d):
        raise ValueError(f"Missing match fields: {set(required) - set(d)}")
    if d[required].isna().any().any():
        raise ValueError("Null values in required match fields")
    d['date'] = pd.to_datetime(d.date, utc=True, format='mixed', errors='raise')
    for side in ['home', 'away']:
        d[f'{side}_team'] = d[f'{side}_team'].map(team_name)
        goals = pd.to_numeric(d[f'{side}_goals'], errors='raise')
        if not (np.isfinite(goals) & (goals >= 0) & (goals % 1 == 0)).all():
            raise ValueError("Invalid goals")
        d[f'{side}_goals'] = goals.astype(int)
    if d.home_team.eq(d.away_team).any():
        raise ValueError("A club cannot play itself")
    if 'status' in d and not d.status.eq('FT').all():
        raise ValueError("Only completed full-time league results are supported")
    if d.duplicated(['season', 'home_team', 'away_team']).any():
        raise ValueError("Duplicate league fixture after team normalization")
    d['day'] = d.date.dt.normalize()
    # Dates have no trusted full-time timestamps. Outcomes enter features next UTC day.
    d['available_at'] = d.day + pd.Timedelta(days=1)
    d['target'] = np.where(d.home_goals > d.away_goals, 0,
                         np.where(d.home_goals == d.away_goals, 1, 2))
    return d.sort_values(['date', 'home_team', 'away_team']).reset_index(drop=True)


def load_data(root=Path('.'), as_of=None):
    root = Path(root)
    pl = pd.read_csv(root / 'premier_league_match_history.csv')
    current = root / 'current_premier_league_results.csv'
    if current.exists():
        pl = pd.concat([pl, pd.read_csv(current)], ignore_index=True)
    pl = validate(pl)
    ch = validate(pd.read_csv(root / 'championship_match_history.csv'))
    # Optional raw match xG is an observed post-match statistic, never a same-match feature.
    raw_path = root / 'EPL_Match_Data_2000_2026.csv'
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        required = {'Year', 'Month', 'Day', 'HomeTeam', 'AwayTeam', 'XGHome', 'XGAway'}
        if required <= set(raw):
            xg = pd.DataFrame({'day': pd.to_datetime(raw[['Year','Month','Day']].rename(
                columns={'Year':'year','Month':'month','Day':'day'}), utc=True),
                'home_team': raw.HomeTeam.map(team_name), 'away_team': raw.AwayTeam.map(team_name),
                'home_xg': pd.to_numeric(raw.XGHome, errors='coerce'),
                'away_xg': pd.to_numeric(raw.XGAway, errors='coerce')})
            pl = pl.merge(xg, on=['day','home_team','away_team'], how='left', validate='one_to_one')
    for col in ['home_xg', 'away_xg']:
        if col not in pl:
            pl[col] = np.nan
        if (pl[col].dropna() < 0).any():
            raise ValueError('Negative xG')
    if as_of is not None:
        cutoff = utc(as_of).normalize()
        pl = pl.loc[pl.available_at <= cutoff].copy()
        ch = ch.loc[ch.available_at <= cutoff].copy()
    for season in set(pl.season) & set(ch.season):
        overlap = clubs(pl[pl.season == season]) & clubs(ch[ch.season == season])
        if overlap:
            raise ValueError(f'Clubs in both divisions in {season}: {sorted(overlap)}')
    return pl.reset_index(drop=True), ch.reset_index(drop=True)


def inventory(root, output):
    """Persist every column's dtype/null count, identifiers, coverage and file hash."""
    entries = []
    for path in sorted(Path(root).glob('*.csv')):
        d = pd.read_csv(path)
        entry = {'file': path.name, 'rows': len(d), 'duplicate_rows': int(d.duplicated().sum()),
                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                 'columns': {c: {'dtype': str(d[c].dtype), 'missing': int(d[c].isna().sum())} for c in d}}
        entry['observation_type'] = ('player_snapshot' if 'snapshot_date' in d else
                                     'prediction_output' if any('probability' in c for c in d) else
                                     'match_records')
        entry['availability_policy'] = ('snapshot date + 1 day; no historical backfill' if 'snapshot_date' in d else
                                        'match results usable next UTC day; source publication/version history unknown')
        for date_col in ['date','Date','kickoff_utc','snapshot_date']:
            if date_col in d:
                dates = pd.to_datetime(d[date_col],utc=True,format='mixed',dayfirst=date_col=='Date',errors='coerce')
                entry['parsed_'+date_col] = {'first':dates.min().isoformat() if dates.notna().any() else None,
                                             'last':dates.max().isoformat() if dates.notna().any() else None,
                                             'invalid':int(dates.isna().sum())}
        for col in ['date','Date','kickoff_utc','season','Season','snapshot_date','edition','league','gender']:
            if col in d:
                values = d[col].dropna().astype(str)
                entry[col] = {'min': values.min() if len(values) else None,
                              'max': values.max() if len(values) else None,
                              'unique_count': int(values.nunique())}
                if values.nunique() <= 30:
                    entry[col]['counts'] = values.value_counts().to_dict()
        for col in ['home_team','away_team','HomeTeam','AwayTeam','club','player_id','fixture_id']:
            if col in d:
                entry[col] = {'unique_count': int(d[col].nunique())}
                if col not in ['player_id','fixture_id']:
                    entry[col]['values'] = sorted(d[col].dropna().astype(str).unique().tolist())
        entries.append(entry)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(entries, indent=2), encoding='utf-8')
    return entries
