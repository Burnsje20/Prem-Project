"""Non-overwriting pre-match records and fixed-policy prospective evaluation.

Checksums detect accidental edits, not deliberate forgery. Local capture times
are not independently notarized timestamps and cannot prove historical availability.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .data import team_name, validate
from .evaluation import metrics, probabilities


PROBABILITY_KEYS = ['home_win_probability', 'draw_probability', 'away_win_probability']


def timestamp(value):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None:
        raise ValueError('Archive timestamps must be known and timezone-aware')
    return t.tz_convert('UTC')


def positive_id(value):
    text = str(value)
    if not text.isascii() or not text.isdigit() or int(text) < 1:
        raise ValueError('A positive integer fixture/player ID is required')
    return str(int(text))


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def source_hashes(paths):
    """Fingerprint explicit local input/code paths; never traverse secrets or caches."""
    return {str(Path(p)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in sorted(set(map(str, paths)))}


def prediction_source_hashes(artifact_dir, data_paths):
    """Shared inference provenance for manual prediction and lineup collection."""
    folder = Path(artifact_dir)
    root = Path(__file__).resolve().parents[1]
    paths = list(folder.glob('*.json')) + list(folder.glob('*.csv'))
    paths += list((root / 'epl_predictor').glob('*.py'))
    paths += [root / name for name in ['hybrid_predictor.py', 'fotmob_client.py', 'poisson_distribution.py']]
    return source_hashes(paths + list(data_paths))


def write_json_new(path, payload):
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(encoded + '\n')
    return path


def validate_capture(payload):
    if payload.get('schema_version') != 1 or payload.get('kind') != 'prospective_forecasts':
        raise ValueError('Unsupported forecast archive schema')
    captured = timestamp(payload['captured_at'])
    records = payload['predictions']
    if not isinstance(records, list) or not records:
        raise ValueError('Cannot archive an empty prediction batch')
    seen = set()
    for row in records:
        if not isinstance(row.get('model'), str) or not row['model'].strip():
            raise ValueError('Archived forecasts require a model name')
        fid = positive_id(row['fixture_id'])
        if fid in seen:
            raise ValueError('Duplicate fixture ID in archive batch')
        seen.add(fid)
        kickoff = timestamp(row['kickoff_utc'])
        if not timestamp(row['data_cutoff']) <= captured < kickoff:
            raise ValueError('Archives require forecasts captured before kickoff and after their data cutoff')
        if timestamp(row['trained_at']) > captured:
            raise ValueError('Model training time is after capture')
        if team_name(row['home_team']) == team_name(row['away_team']):
            raise ValueError('Archive clubs must differ')
        probabilities([[row[k] for k in PROBABILITY_KEYS]])
        snapshot = (row.get('live_lineup_context') or {}).get('snapshot')
        if snapshot is not None:
            if not timestamp(snapshot['fetched_at']) <= captured < timestamp(snapshot['kickoff_utc']):
                raise ValueError('Lineup capture time is invalid')
            if (snapshot['status'] != 'prematch' or positive_id(snapshot['fixture_id']) != fid or
                    timestamp(snapshot['kickoff_utc']) != kickoff):
                raise ValueError('Lineup does not belong to the archived pre-match fixture')
            for side in ['home', 'away']:
                if team_name(snapshot[side]['team']) != team_name(row[f'{side}_team']):
                    raise ValueError('Archived lineup team mismatch')


def archive_predictions(predictions, directory, inputs):
    """Capture now, never accept a user-supplied backdated archive time."""
    if source_hashes(inputs) != inputs:
        raise ValueError('Source files changed during prediction; rerun before archiving')
    captured = datetime.now(timezone.utc)
    payload = {'schema_version': 1, 'kind': 'prospective_forecasts',
               'captured_at': captured.isoformat(), 'run_id': uuid4().hex,
               'source_hashes': inputs, 'predictions': predictions}
    validate_capture(payload)
    payload['content_sha256'] = digest(payload)
    filename = f"forecast_{captured.strftime('%Y%m%dT%H%M%S%fZ')}_{payload['run_id']}.json"
    return write_json_new(Path(directory) / filename, payload)


def load_archive(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    checksum = payload.pop('content_sha256', None)
    if checksum != digest(payload):
        raise ValueError(f'Archive checksum mismatch: {path}')
    validate_capture(payload)
    payload['content_sha256'] = checksum
    return payload


def evaluate_archive(directory, results):
    """Latest archived pre-kickoff forecast per fixture; no result-based selection.

    Only next-UTC-day-available FT results are scored, with exact ID/club/kickoff
    agreement. Changed schedules are flagged, not silently joined by team names.
    """
    paths = sorted(Path(directory).glob('forecast_*.json'))
    if not paths:
        raise ValueError('No forecast archives found')
    now = timestamp(datetime.now(timezone.utc))
    selected = {}
    for path in paths:
        archive = load_archive(path)
        captured = timestamp(archive['captured_at'])
        if captured > now:
            raise ValueError('Archive capture time is in the future')
        for row in archive['predictions']:
            fid = positive_id(row['fixture_id'])
            key = (captured, archive['run_id'])
            if fid not in selected or key > selected[fid][0]:
                selected[fid] = (key, row, str(path))
    if 'fixture_id' not in results:
        raise ValueError('Prospective evaluation requires result fixture_id values')
    d = validate(results)
    d['fixture_id'] = d.fixture_id.map(positive_id)
    if d.fixture_id.duplicated().any():
        raise ValueError('Duplicate result fixture IDs')
    d = d[d.available_at <= now.normalize()].set_index('fixture_id')
    scored, pending, mismatched = [], [], []
    for fid, (key, row, path) in sorted(selected.items()):
        if fid not in d.index:
            pending.append(fid)
            continue
        actual = d.loc[fid]
        if (team_name(row['home_team']) != actual.home_team or
                team_name(row['away_team']) != actual.away_team or
                timestamp(row['kickoff_utc']) != actual.date):
            mismatched.append(fid)
            continue
        scored.append({'fixture_id': fid, 'archive': path, 'captured_at': key[0].isoformat(),
                       'model': row['model'], 'target': int(actual.target),
                       'probabilities': [row[k] for k in PROBABILITY_KEYS]})
    scores = metrics([r['target'] for r in scored], [r['probabilities'] for r in scored]) if scored else None
    return {'selection_policy': 'latest captured pre-kickoff forecast per fixture; run_id breaks timestamp ties',
            'evaluated_at': now.isoformat(), 'archive_count': len(paths),
            'forecast_count': len(selected), 'metrics': scores, 'scored': scored,
            'pending_fixture_ids': pending, 'mismatched_fixture_ids': mismatched,
            'caveat': 'Local timestamps/checksums are not independent proof of availability; no model selection is performed'}
