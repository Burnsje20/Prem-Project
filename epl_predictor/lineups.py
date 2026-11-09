"""Explicit opt-in live context. The validated forecast remains untouched."""
from datetime import datetime, timezone

import requests

from .data import team_name, utc
from .players import match_starting_xi_talent, validate_player_bridge


def add_live_lineup_context(predictions, client, ea=None, bridge=None, now=None):
    """Enrich forecast records without training or changing their probabilities.

    Capture time is evidence of availability, not the provider publication time.
    Today's lineups must never be retroactively used for a historical prediction.
    """
    validate_player_bridge(bridge)
    cache = {}
    stopped = None
    for result in predictions:
        context = {'status': 'unavailable', 'used_in_model': False,
                   'reason': None, 'snapshot': None,
                   'model_policy': 'No validated talent-to-goals coefficient; forecasts unchanged'}
        result['live_lineup_context'] = context
        current = utc(now or datetime.now(timezone.utc))
        if current >= utc(result['kickoff_utc']):
            context['reason'] = 'Kickoff has passed; live lineup lookup would leak into a historical forecast'
            continue
        fixture_id = result.get('fixture_id')
        if fixture_id is None:
            context['reason'] = 'No FotMob fixture_id supplied'
            continue
        if stopped:
            context['reason'] = stopped
            continue
        try:
            if fixture_id not in cache:
                cache[fixture_id] = client.fetch_lineup(fixture_id)
            snapshot = cache[fixture_id]
            fetched = utc(snapshot['fetched_at'])
            if (snapshot['status'] != 'prematch' or fetched >= utc(result['kickoff_utc']) or
                    int(snapshot['fixture_id']) != int(fixture_id) or
                    utc(snapshot['kickoff_utc']) != utc(result['kickoff_utc'])):
                raise ValueError('Fixture time/identity changed or lineup is not pre-match; refresh fixtures')
            for side in ['home', 'away']:
                if team_name(snapshot[side]['team']) != result[f'{side}_team']:
                    raise ValueError('Lineup clubs do not match the forecast')
            context['snapshot'] = snapshot
            for side in ['home', 'away']:
                context[side] = match_starting_xi_talent(
                    result[f'{side}_team'], snapshot[side]['starters'], ea, fetched, bridge)
            has_starters = any(snapshot[s]['starters'] for s in ['home', 'away'])
            context['status'] = 'available' if has_starters else 'unavailable'
            context['reason'] = None if has_starters else 'Provider has not supplied starters'
        except requests.HTTPError as error:
            context['reason'] = f'FotMob HTTP error: {error.response.status_code if error.response is not None else "unknown"}'
            if error.response is not None and error.response.status_code in {401, 403, 429}:
                stopped = 'Live lookup stopped after provider access/rate-limit response; retry later'
        except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as error:
            context['status'] = 'unavailable'
            context['snapshot'] = None
            context.pop('home', None)
            context.pop('away', None)
            context['reason'] = f'Lineup unavailable: {type(error).__name__}: {error}'
    return predictions
