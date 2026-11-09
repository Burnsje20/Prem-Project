"""One-shot near-kickoff lineup collection; no background process or automatic approvals."""
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from fotmob_client import FotMobClient
from .data import team_name
from .lineups import add_live_lineup_context
from .player_review import create_review
from .players import validate_player_bridge
from .predictor import Predictor
from .prospective import (
    archive_predictions, load_archive, positive_id, prediction_source_hashes,
    timestamp, write_json_new,
)


def select_collection_fixtures(fixtures, now, within_minutes=90, refresh_minutes=10, recent=None):
    """Choose scheduled FotMob fixtures by wall-clock time, not model cutoff."""
    if not 1 <= within_minutes <= 1440 or not 1 <= refresh_minutes <= within_minutes:
        raise ValueError('Use a 1–1440 minute window and refresh interval between 1 and the window')
    required = {'fixture_id', 'kickoff_utc', 'home_team', 'away_team', 'status', 'source'}
    if not required <= set(fixtures):
        raise ValueError(f'Collection fixtures missing: {sorted(required - set(fixtures))}')
    d = fixtures.copy().reset_index(drop=True)
    d['fixture_id'] = d.fixture_id.map(positive_id)
    if d.fixture_id.duplicated().any():
        raise ValueError('Duplicate collection fixture IDs')
    d['kickoff_utc'] = d.kickoff_utc.map(timestamp)
    for side in ['home', 'away']:
        d[f'{side}_team'] = d[f'{side}_team'].map(team_name)
    if d.home_team.eq(d.away_team).any():
        raise ValueError('Collection fixture clubs must differ')
    now, recent = timestamp(now), recent or {}
    due, skipped, next_times = [], [], []
    for i, row in d.iterrows():
        reason = None
        if str(row.source).lower() != 'fotmob':
            reason = 'unsupported_source'
        elif str(row.status).strip().lower() not in {'scheduled', 'ns', 'not started'}:
            reason = 'not_scheduled'
        elif row.kickoff_utc <= now:
            reason = 'kickoff_passed'
        elif row.kickoff_utc - pd.Timedelta(minutes=within_minutes) > now:
            reason = 'outside_window'
            next_times.append(row.kickoff_utc - pd.Timedelta(minutes=within_minutes))
        elif row.fixture_id in recent:
            retry_at = timestamp(recent[row.fixture_id]) + pd.Timedelta(minutes=refresh_minutes)
            if retry_at > now:
                reason = 'refresh_cooldown'
                if retry_at < row.kickoff_utc:
                    next_times.append(retry_at)
        if reason:
            skipped.append({'fixture_id': row.fixture_id, 'reason': reason})
        else:
            due.append(i)
    return d.loc[due].sort_values(['kickoff_utc', 'fixture_id']).reset_index(drop=True), skipped, (
        min(next_times).isoformat() if next_times else None)


def recent_lineup_captures(archive_dir, now):
    """An archived lineup attempt, including provider failures, starts the cooldown."""
    recent = {}
    for path in sorted(Path(archive_dir).glob('forecast_*.json')):
        archive = load_archive(path)
        captured = timestamp(archive['captured_at'])
        if captured > timestamp(now):
            raise ValueError('Existing archive has a future capture time')
        for row in archive['predictions']:
            if 'live_lineup_context' not in row:
                continue
            fid = positive_id(row['fixture_id'])
            if fid not in recent or captured > recent[fid]:
                recent[fid] = captured
    return recent


def collect_lineups(fixtures_path='upcoming_premier_league_fixtures.csv',
                    artifact_dir='artifacts/hybrid_v1', players_path='players.csv',
                    player_map_path=None, archive_dir='archives/prospective',
                    output_dir='reports/lineup_collection', within_minutes=90, refresh_minutes=10):
    """Run once: select, predict, fetch, archive and emit an unapproved review queue.

    No eligible fixtures means no model loading and no provider calls. This does
    not schedule itself. Repeated invocations observe a persisted archive cooldown.
    """
    started = datetime.now(timezone.utc)
    inputs_paths = [Path(fixtures_path)]
    if players_path and Path(players_path).exists():
        inputs_paths.append(Path(players_path))
    if player_map_path:
        inputs_paths.append(Path(player_map_path))
    inputs = prediction_source_hashes(artifact_dir, inputs_paths)
    fixtures = pd.read_csv(fixtures_path, dtype={'fixture_id': str})
    recent = recent_lineup_captures(archive_dir, started)
    due, skipped, next_at = select_collection_fixtures(
        fixtures, started, within_minutes, refresh_minutes, recent)
    report = {'schema_version': 1, 'started_at': started.isoformat(), 'status': 'idle',
              'within_minutes': within_minutes, 'refresh_minutes': refresh_minutes,
              'eligible_fixture_ids': due.fixture_id.tolist(), 'skipped': skipped,
              'next_collection_at_utc': next_at, 'archive': None, 'review': None,
              'archived_count': 0, 'lineups_available': 0, 'unresolved_players': 0,
              'lineup_statuses': [], 'source_hashes': inputs,
              'forecast_policy': 'Lineup context only; probabilities unchanged'}
    if not due.empty:
        bridge = pd.read_csv(player_map_path, dtype=str) if player_map_path else None
        validate_player_bridge(bridge)
        ea = pd.read_csv(players_path) if players_path and Path(players_path).exists() else None
        predictions = Predictor(artifact_dir).predict(due, ea)
        client = FotMobClient()
        try:
            add_live_lineup_context(predictions, client, ea, bridge)
        finally:
            client.session.close()
        finished = timestamp(datetime.now(timezone.utc))
        # A long request batch may cross kickoff. Never backdate the archive to
        # the start of the run to make those predictions appear prospective.
        safe = []
        for row in predictions:
            if timestamp(row['kickoff_utc']) <= finished:
                report['skipped'].append({'fixture_id': str(row['fixture_id']), 'reason': 'kickoff_passed_during_collection'})
            else:
                safe.append(row)
        if safe:
            path = archive_predictions(safe, archive_dir, inputs)
            report['archive'] = str(path)
            review_path = Path(output_dir) / f'player_review_{path.stem}.json'
            create_review([path], review_path)
            report['review'] = str(review_path)
            report['unresolved_players'] = len(json.loads(review_path.read_text(encoding='utf-8'))['decisions'])
            report['archived_count'] = len(safe)
            report['status'] = 'collected'
            for row in safe:
                context = row['live_lineup_context']
                report['lineup_statuses'].append({'fixture_id': str(row['fixture_id']),
                    'status': context['status'], 'reason': context['reason']})
                report['lineups_available'] += int(context['status'] == 'available')
            captured = timestamp(load_archive(path)['captured_at'])
            retries = [captured + pd.Timedelta(minutes=refresh_minutes)
                       for row in safe if captured + pd.Timedelta(minutes=refresh_minutes) < timestamp(row['kickoff_utc'])]
            if next_at:
                retries.append(timestamp(next_at))
            report['next_collection_at_utc'] = min(retries).isoformat() if retries else None
        else:
            report['status'] = 'expired_during_collection'
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    filename = f"collection_{started.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex}.json"
    report_path = write_json_new(Path(output_dir) / filename, report)
    return report, report_path
