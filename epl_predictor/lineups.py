"""Explicit opt-in live context and validated starter form adjustment."""
from datetime import datetime, timezone

import numpy as np
import requests

from .data import team_name, utc
from .goals import markets, outcomes, reconcile, score_matrix
from .players import match_starting_xi_talent, starting_xi_fotmob_form, validate_player_bridge


def adjust_prediction_with_starter_form(prediction, home_starters, away_starters,
                                       appearances, rho=0.0, beta=0.1675):
    """Adjust goal rates and outcome probabilities based on FotMob starter form difference.

    lambda_adj = lambda_base * exp(beta * (home_form - away_form))
    mu_adj = mu_base * exp(-beta * (home_form - away_form))
    """
    cutoff = utc(prediction['kickoff_utc'])
    home_form = starting_xi_fotmob_form(home_starters, appearances, cutoff)
    away_form = starting_xi_fotmob_form(away_starters, appearances, cutoff)

    form_diff = home_form['mean_starter_form'] - away_form['mean_starter_form']

    if 'baseline_forecast' not in prediction:
        prediction['baseline_forecast'] = {
            'home_win_probability': prediction['home_win_probability'],
            'draw_probability': prediction['draw_probability'],
            'away_win_probability': prediction['away_win_probability'],
            'expected_home_goals': prediction['expected_home_goals'],
            'expected_away_goals': prediction['expected_away_goals'],
            'most_likely_score': prediction['most_likely_score'],
        }

    lam_base = prediction['expected_home_goals']
    mu_base = prediction['expected_away_goals']

    lam_adj = float(np.clip(lam_base * np.exp(beta * form_diff), 0.05, 10.0))
    mu_adj = float(np.clip(mu_base * np.exp(-beta * form_diff), 0.05, 10.0))

    matrix = score_matrix(lam_adj, mu_adj, rho=rho)
    coherent = reconcile(matrix, outcomes(matrix))
    stats = markets(coherent)

    prediction.update(stats)
    prediction['starter_form_adjustment'] = {
        'applied': True,
        'beta': beta,
        'home_starter_form': home_form['mean_starter_form'],
        'away_starter_form': away_form['mean_starter_form'],
        'form_diff': round(form_diff, 4),
        'home_form_details': home_form,
        'away_form_details': away_form
    }
    prediction['player_features_used_in_model'] = True
    prediction['model'] = f"{prediction['model']}+starter_form"
    return prediction


def add_live_lineup_context(predictions, client, ea=None, bridge=None, now=None,
                            appearances=None, adjust_probabilities=False, beta=0.1675, rho=0.0):
    """Enrich forecast records with live lineups and optional FotMob starter form adjustment.

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

            if appearances is not None and has_starters:
                for side in ['home', 'away']:
                    if snapshot[side]['starters']:
                        context[side]['starter_form'] = starting_xi_fotmob_form(
                            snapshot[side]['starters'], appearances, fetched)

            if adjust_probabilities and appearances is not None and has_starters:
                adjust_prediction_with_starter_form(
                    result, snapshot['home']['starters'], snapshot['away']['starters'],
                    appearances, rho=rho, beta=beta)
                context['used_in_model'] = True
                context['model_policy'] = f'Adjusted by FotMob recent starter form (beta={beta})'
            elif has_starters and appearances is not None:
                context['used_in_model'] = False
                context['model_policy'] = 'FotMob starter form attached as context; probabilities unchanged'
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
