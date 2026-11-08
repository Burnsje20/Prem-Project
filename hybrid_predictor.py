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
    args = parser.parse_args()
    if args.command == 'audit':
        entries = inventory(Path('.'),args.output)
        print(f'Audited {len(entries)} CSV files: {args.output}')
    elif args.command == 'train':
        from epl_predictor.workflow import train
        train(config_path=args.config,artifacts=args.artifacts,reports=args.reports)
    else:
        from epl_predictor.predictor import Predictor
        model = Predictor(args.artifacts)
        if bool(args.home) != bool(args.away):
            parser.error('Supply both --home and --away')
        fixtures = (pd.DataFrame([{'home_team':args.home,'away_team':args.away,
                                  'kickoff_utc':args.date or model.manifest['cutoff']}])
                    if args.home else pd.read_csv(args.fixtures))
        players = pd.read_csv(args.players) if args.players.exists() else None
        appearances = pd.read_csv(args.appearances) if args.appearances else None
        predictions = model.predict(fixtures,players,appearances)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(predictions,indent=2,allow_nan=False),encoding='utf-8')
        for result in predictions:
            print(f"{result['home_team']} vs {result['away_team']}: "
                  f"H {result['home_win_probability']:.1%} / D {result['draw_probability']:.1%} / "
                  f"A {result['away_win_probability']:.1%}; score {result['most_likely_score']}")
        print(f"Model: {model.manifest['selected_model']}; saved {len(predictions)} forecasts to {args.output}")


if __name__ == '__main__':
    main()
