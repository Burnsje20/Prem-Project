"""Collect and validate dated player appearances from completed FotMob match feeds."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Optional

import pandas as pd

from fotmob_client import FotMobClient
from .players import validate_appearances


def collect_match_appearances(
    results_path: str | Path = 'current_premier_league_results.csv',
    output_path: str | Path = 'fotmob_player_appearances.csv',
    client: Optional[FotMobClient] = None,
    limit: Optional[int] = None,
    delay: float = 0.2,
) -> dict[str, Any]:
    """Collect player appearances from completed matches into a validated CSV.

    Preserves existing collected matches and only queries new or missing fixtures.
    Explicitly tracks retrieved_at and ensures outputs conform strictly to
    validate_appearances.
    """
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f'Results file not found: {results_path}')
    output_path = Path(output_path)

    results = pd.read_csv(results_path)
    if 'fixture_id' not in results:
        raise ValueError("Results file missing 'fixture_id' column")

    # Filter to finished matches
    finished = results[results.get('status', 'FT').str.upper().isin(['FT', 'AET', 'FINISHED'])].copy()
    if finished.empty:
        finished = results.copy()

    # Load existing appearances if available
    existing_appearances = pd.DataFrame()
    collected_match_ids = set()
    if output_path.exists():
        try:
            existing_appearances = pd.read_csv(output_path)
            if not existing_appearances.empty and 'match_id' in existing_appearances:
                collected_match_ids = set(existing_appearances['match_id'].astype(str))
        except Exception:
            existing_appearances = pd.DataFrame()

    to_collect = []
    for fid in finished['fixture_id']:
        sfid = str(fid)
        if sfid not in collected_match_ids and sfid not in to_collect:
            to_collect.append(sfid)

    if limit is not None and limit > 0:
        to_collect = to_collect[:limit]

    new_appearances = []
    skipped = []
    close_client = False
    if client is None and to_collect:
        client = FotMobClient()
        close_client = True

    started = datetime.now(timezone.utc)
    try:
        for idx, match_id in enumerate(to_collect):
            try:
                records = client.fetch_match_appearances(int(match_id))
                new_appearances.extend(records)
            except Exception as e:
                skipped.append({'fixture_id': match_id, 'reason': str(e)})

            if delay > 0 and idx < len(to_collect) - 1:
                time.sleep(delay)
    finally:
        if close_client and client is not None:
            client.session.close()

    # Merge existing and newly collected appearances
    if new_appearances:
        new_df = pd.DataFrame(new_appearances)
        combined = pd.concat([existing_appearances, new_df], ignore_index=True) if not existing_appearances.empty else new_df
    else:
        combined = existing_appearances

    if not combined.empty:
        # Validate against strict player appearance contracts
        validated = validate_appearances(combined)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Drop duplicates on (player_id, match_id) just in case
        validated = validated.drop_duplicates(subset=['player_id', 'match_id'], keep='last')
        validated.to_csv(output_path, index=False)
        total_records = len(validated)
        total_matches = int(validated['match_id'].nunique())
    else:
        total_records = 0
        total_matches = 0

    return {
        'schema_version': 1,
        'started_at': started.isoformat(),
        'finished_at': datetime.now(timezone.utc).isoformat(),
        'total_appearances': total_records,
        'total_matches': total_matches,
        'newly_collected_matches': len(to_collect) - len(skipped),
        'attempted_matches': len(to_collect),
        'skipped_matches': skipped,
        'output_path': str(output_path),
    }
