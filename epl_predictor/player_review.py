"""Turn unresolved lineup identities into an explicit, human-reviewed crosswalk."""
import csv
import json
from pathlib import Path

import pandas as pd

from .players import validate_player_bridge
from .prospective import load_archive, positive_id, source_hashes, write_json_new


def create_review(prediction_paths, output):
    rows = {}
    for path in prediction_paths:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        predictions = data if isinstance(data, list) else load_archive(path)['predictions']
        for prediction in predictions:
            context = prediction.get('live_lineup_context') or {}
            if context.get('status') != 'available':
                continue
            for side in ['home', 'away']:
                for player in context.get(side, {}).get('unmatched', []):
                    fid = positive_id(player['player_id'])
                    row = rows.setdefault(fid, {'fotmob_id': fid, 'names': set(), 'clubs': set(),
                        'fixture_ids': set(), 'candidate_ea_ids': set(), 'ea_id': None,
                        'approved': False, 'review_note': ''})
                    if player.get('name'):
                        row['names'].add(player['name'])
                    row['clubs'].add(prediction[f'{side}_team'])
                    row['fixture_ids'].add(positive_id(prediction['fixture_id']))
                    row['candidate_ea_ids'].update(positive_id(c) for c in player.get('candidate_ea_ids', []))
    decisions = []
    for fid in sorted(rows, key=int):
        row = rows[fid]
        decisions.append({k: sorted(v) if isinstance(v, set) else v for k, v in row.items()})
    return write_json_new(output, {'schema_version': 1, 'kind': 'player_id_review',
        'instructions': 'Verify identities independently, set ea_id, approved=true and review_note. Candidates are not verified. Unapproved rows are never exported.',
        'source_hashes': source_hashes(prediction_paths), 'decisions': decisions})


def export_reviewed_bridge(review_path, output):
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if review.get('schema_version') != 1 or review.get('kind') != 'player_id_review':
        raise ValueError('Unsupported player review schema')
    rows, seen = [], set()
    for decision in review['decisions']:
        fid = positive_id(decision['fotmob_id'])
        if fid in seen:
            raise ValueError('Duplicate FotMob identity in review')
        seen.add(fid)
        if type(decision.get('approved')) is not bool:
            raise ValueError('approved must be a JSON boolean')
        if not decision['approved']:
            continue
        if not isinstance(decision.get('review_note'), str) or not decision['review_note'].strip():
            raise ValueError('Approved identities require a review_note explaining verification')
        rows.append({'ea_id': positive_id(decision['ea_id']), 'fotmob_id': fid})
    if not rows:
        raise ValueError('No approved player identities; nothing exported')
    validate_player_bridge(pd.DataFrame(rows))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['ea_id', 'fotmob_id'])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
