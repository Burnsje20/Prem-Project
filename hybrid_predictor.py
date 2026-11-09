"""Audit, train/evaluate, and predict with the saved Premier League model."""
import argparse
import json
from pathlib import Path

import pandas as pd

from epl_predictor.data import inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    audit = commands.add_parser('audit')
    audit.add_argument('--output',type=Path,default=Path('reports/data_inventory.json'))
    training = commands.add_parser('train')
    training.add_argument('--config',type=Path,default=Path('hybrid_config.json'))
    training.add_argument('--artifacts',type=Path,default=Path('artifacts/hybrid_v1'))
    training.add_argument('--reports',type=Path,default=Path('reports'))
    predict = commands.add_parser('predict')
    predict.add_argument('--artifacts',type=Path,default=Path('artifacts/hybrid_v1'))
    predict.add_argument('--fixtures',type=Path,default=Path('upcoming_premier_league_fixtures.csv'))
    predict.add_argument('--home')
    predict.add_argument('--away')
    predict.add_argument('--date',help='Kickoff in ISO format; default is saved model cutoff')
    predict.add_argument('--output',type=Path,default=Path('hybrid_predictions.json'))
    predict.add_argument('--players',type=Path,default=Path('players.csv'))
    predict.add_argument('--appearances',type=Path,help='Optional validated FotMob player-match CSV')
    predict.add_argument('--filter-future',action='store_true',help='Filter fixtures to kickoff >= cutoff')
    predict.add_argument('--use-lineups',action='store_true',help='Fetch live FotMob lineup context; does not adjust probabilities')
    predict.add_argument('--fixture-id',type=int,help='FotMob match ID when using --home/--away')
    predict.add_argument('--player-id-map',type=Path,help='Reviewed CSV crosswalk with ea_id,fotmob_id columns')
    predict.add_argument('--archive-dir',type=Path,help='Also save a non-overwriting, timestamped pre-match archive')
    review = commands.add_parser('review-players',help='Create an unapproved identity review JSON from forecasts/archives')
    review.add_argument('--predictions',type=Path,nargs='+',required=True)
    review.add_argument('--output',type=Path,required=True)
    export = commands.add_parser('export-player-map',help='Export explicitly approved identities to a new CSV')
    export.add_argument('--review',type=Path,required=True)
    export.add_argument('--output',type=Path,required=True)
    evaluate = commands.add_parser('evaluate-live',help='Score archived pre-match forecasts against finished results')
    evaluate.add_argument('--archive-dir',type=Path,required=True)
    evaluate.add_argument('--results',type=Path,default=Path('current_premier_league_results.csv'))
    evaluate.add_argument('--output',type=Path,required=True)
    collect = commands.add_parser('collect-lineups',help='Collect near-kickoff lineups once, archive forecasts and prepare identity review')
    collect.add_argument('--fixtures',type=Path,default=Path('upcoming_premier_league_fixtures.csv'))
    collect.add_argument('--artifacts',type=Path,default=Path('artifacts/hybrid_v1'))
    collect.add_argument('--players',type=Path,default=Path('players.csv'))
    collect.add_argument('--player-id-map',type=Path)
    collect.add_argument('--archive-dir',type=Path,default=Path('archives/prospective'))
    collect.add_argument('--output-dir',type=Path,default=Path('reports/lineup_collection'))
    collect.add_argument('--within-minutes',type=int,default=90)
    collect.add_argument('--refresh-minutes',type=int,default=10)
    uncertainty = commands.add_parser('uncertainty',help='Show paired date-block bootstrap uncertainty')
    uncertainty.add_argument('--reports',type=Path,default=Path('reports'))
    args = parser.parse_args()
    if args.command == 'audit':
        entries = inventory(Path('.'),args.output)
        print(f'Audited {len(entries)} CSV files: {args.output}')
    elif args.command == 'train':
        from epl_predictor.workflow import train
        train(config_path=args.config,artifacts=args.artifacts,reports=args.reports)
    elif args.command == 'uncertainty':
        path = args.reports / 'model_uncertainty.csv'
        if not path.exists():
            parser.error(f'Uncertainty report not found: {path}')
        d = pd.read_csv(path)
        print('Test Period Paired Date-Block Bootstrap Uncertainty (vs Selected Dixon-Coles):')
        cols = ['comparison_model','point_difference','std_error','ci_lower_95','ci_upper_95','statistically_significant']
        print(d[cols].to_string(index=False))
    elif args.command == 'review-players':
        from epl_predictor.player_review import create_review
        path = create_review(args.predictions,args.output)
        print(f'Created unapproved player identity review: {path}')
    elif args.command == 'export-player-map':
        from epl_predictor.player_review import export_reviewed_bridge
        count = export_reviewed_bridge(args.review,args.output)
        print(f'Exported {count} reviewed player mappings: {args.output}')
    elif args.command == 'evaluate-live':
        from epl_predictor.prospective import evaluate_archive, write_json_new
        report = evaluate_archive(args.archive_dir,pd.read_csv(args.results))
        write_json_new(args.output,report)
        print(f"Scored {len(report['scored'])} fixtures; pending {len(report['pending_fixture_ids'])}; "
              f"identity/schedule mismatches {len(report['mismatched_fixture_ids'])}. Saved {args.output}")
        if report['metrics']:
            print(json.dumps(report['metrics'],indent=2))
    elif args.command == 'collect-lineups':
        from epl_predictor.collection import collect_lineups
        report, path = collect_lineups(args.fixtures,args.artifacts,args.players,args.player_id_map,
            args.archive_dir,args.output_dir,args.within_minutes,args.refresh_minutes)
        print(f"Collection {report['status']}: {report['archived_count']} archived, "
              f"{report['lineups_available']} with starters, {report['unresolved_players']} players to review.")
        if report['next_collection_at_utc']:
            print(f"Next collection window/retry (UTC): {report['next_collection_at_utc']}")
        print(f'Collection report: {path}')
    else:
        from epl_predictor.predictor import Predictor
        inputs = {}
        if args.archive_dir:
            from epl_predictor.prospective import prediction_source_hashes
            paths = [p for p in [args.players,args.appearances,args.player_id_map] if p and p.exists()]
            if not args.home:
                paths.append(args.fixtures)
            inputs = prediction_source_hashes(args.artifacts,paths)
        model = Predictor(args.artifacts)
        if bool(args.home) != bool(args.away):
            parser.error('Supply both --home and --away')
        if args.fixture_id is not None and (not args.home or args.fixture_id < 1):
            parser.error('--fixture-id requires --home/--away and a positive integer')
        if args.use_lineups and args.home and (args.fixture_id is None or args.date is None):
            parser.error('Live single-match lookup requires --fixture-id and --date')
        if args.player_id_map and not args.use_lineups:
            parser.error('--player-id-map requires --use-lineups')
        fixtures = (pd.DataFrame([{'home_team':args.home,'away_team':args.away,
                                  'fixture_id':args.fixture_id,
                                  'kickoff_utc':args.date or model.manifest['cutoff']}])
                    if args.home else pd.read_csv(args.fixtures))
        players = pd.read_csv(args.players) if args.players.exists() else None
        appearances = pd.read_csv(args.appearances) if args.appearances else None
        predictions = model.predict(fixtures,players,appearances,filter_future=args.filter_future)
        if args.use_lineups:
            from fotmob_client import FotMobClient
            from epl_predictor.lineups import add_live_lineup_context
            bridge = pd.read_csv(args.player_id_map,dtype=str) if args.player_id_map else None
            client = FotMobClient()
            try:
                add_live_lineup_context(predictions,client,players,bridge)
            finally:
                client.session.close()
            print('Live lineups are experimental context only; forecast probabilities are unchanged.')
        if args.archive_dir:
            from epl_predictor.prospective import archive_predictions
            path = archive_predictions(predictions,args.archive_dir,inputs)
            print(f'Archived pre-match forecasts: {path}')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(predictions,indent=2,allow_nan=False),encoding='utf-8')
        for result in predictions:
            print(f"{result['home_team']} vs {result['away_team']}: "
                  f"H {result['home_win_probability']:.1%} / D {result['draw_probability']:.1%} / "
                  f"A {result['away_win_probability']:.1%}; score {result['most_likely_score']}")
        print(f"Model: {model.manifest['selected_model']}; saved {len(predictions)} forecasts to {args.output}")


if __name__ == '__main__':
    main()
