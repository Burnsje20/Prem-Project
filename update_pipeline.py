"""Unified automated update pipeline for local execution and GitHub Actions.

Steps executed:
1. Refresh upcoming 30-day Premier League fixtures from FotMob.
2. Incrementally collect appearances/ratings for newly finished matches.
3. Check and capture live starting lineups for matches near kickoff (<90 mins).
4. Rebuild the interactive dashboard at docs/index.html for GitHub Pages.
"""
from datetime import date
from pathlib import Path
import sys

import pandas as pd

from fotmob_client import FotMobClient
from fetch_upcoming_fixtures import OUTPUT_PATH as FIXTURES_PATH, OUTPUT_COLUMNS
from epl_predictor.appearances import collect_match_appearances
from epl_predictor.collection import collect_lineups
from epl_predictor.dashboard import build_dashboard


def run_pipeline() -> dict:
    print("=" * 60)
    print("STARTING PREMIER LEAGUE PREDICTOR AUTOMATED UPDATE PIPELINE")
    print("=" * 60)

    # 1. Update upcoming fixtures
    print("\n[Step 1/4] Refreshing upcoming Premier League fixtures...")
    client = FotMobClient()
    try:
        fixtures = client.upcoming_fixtures(date.today(), days=30)
        fixtures_df = pd.DataFrame(
            [{'fixture_id': f.fixture_id, 'kickoff_utc': f.kickoff_utc.isoformat(),
              'season': f.season, 'home_team': f.home_team, 'away_team': f.away_team,
              'status': f.status, 'source': f.source}
             for f in fixtures],
            columns=OUTPUT_COLUMNS
        )
        if not fixtures_df.empty:
            fixtures_df.to_csv(FIXTURES_PATH, index=False)
            print(f"-> Successfully saved {len(fixtures_df)} upcoming fixtures to {FIXTURES_PATH}")
        else:
            print("-> No fixtures returned; preserving existing upcoming_premier_league_fixtures.csv")
    except Exception as e:
        print(f"-> Warning: Could not refresh upcoming fixtures ({e}); keeping current cache.")
    finally:
        client.session.close()

    # 2. Collect completed appearances
    print("\n[Step 2/4] Incrementally collecting player appearances from finished matches...")
    appearances_rep = collect_match_appearances(
        results_path='current_premier_league_results.csv',
        output_path='fotmob_player_appearances.csv',
        delay=0.1
    )
    print(f"-> Newly collected: {appearances_rep['newly_collected_matches']} matches. "
          f"Total appearances: {appearances_rep['total_appearances']} across {appearances_rep['total_matches']} matches.")

    # 3. Check for imminent lineups near kickoff
    print("\n[Step 3/4] Checking for near-kickoff starting lineups (<90 mins)...")
    collection_rep, col_path = collect_lineups(
        fixtures_path=str(FIXTURES_PATH),
        artifact_dir='artifacts/hybrid_v1',
        players_path='players.csv',
        archive_dir='archives/prospective',
        output_dir='reports/lineup_collection',
        appearances_path='fotmob_player_appearances.csv',
        adjust_probabilities=True
    )
    print(f"-> Collection status: {collection_rep['status']}. "
          f"Archived: {collection_rep['archived_count']} fixtures, "
          f"Starters available: {collection_rep['lineups_available']}.")
    if collection_rep.get('next_collection_at_utc'):
        print(f"-> Next lineup collection window (UTC): {collection_rep['next_collection_at_utc']}")

    # 4. Rebuild interactive dashboard for GitHub Pages
    print("\n[Step 4/4] Building interactive matchday dashboard for GitHub Pages...")
    docs_path = build_dashboard(output_path='docs/index.html')
    reports_path = build_dashboard(output_path='reports/matchday_12_dashboard.html')
    print(f"-> Successfully built dashboard at {docs_path} and {reports_path}")

    print("\n" + "=" * 60)
    print("PIPELINE UPDATE COMPLETE")
    print("=" * 60)

    return {
        'fixtures_count': len(fixtures_df) if 'fixtures_df' in locals() and not fixtures_df.empty else None,
        'new_appearances_matches': appearances_rep['newly_collected_matches'],
        'total_appearances': appearances_rep['total_appearances'],
        'collection_status': collection_rep['status'],
        'dashboard_path': str(docs_path)
    }


if __name__ == '__main__':
    run_pipeline()
