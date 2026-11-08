# Testing the Poisson predictor

Run the offline replay against `current_premier_league_results.csv`:

```powershell
python backtest_poisson.py
```

The command writes `poisson_backtest_predictions.csv` (predictions and actual
results for each method) and `poisson_backtest_summary.json` (aggregate metrics).
It does not refresh data or overwrite the upcoming prediction CSV.

Three methods are compared on the same completed matches:

- `historical_only`: withholds every current-season result, including from
  the promotion adjustment and league goal rate. Uses historical promotion factors.
- `walk_forward`: predicts one UTC date at a time using only results from earlier
  dates. Both team form and the promotion adjustment are recalculated at that cutoff.
  Same-day games are excluded because the cache lacks final-whistle times.
- `league_average`: gives every match the prior PL season's average home/away
  goal rates, providing a simple comparison without team-specific ratings.

The historical files are restricted to seasons before the test season. Predictions
are made before attaching actual scores. The match-level output records the cutoff,
number of current results used, latest included kickoff, and current promotion weight.

Result accuracy measures the most probable H/D/A outcome. Exact-score accuracy
measures the most probable scoreline. Lower log loss and Brier score indicate better
probabilities; Brier is summed over three outcomes (range 0–2). Goal MAE is the
average absolute error per team. A better log loss can accompany lower accuracy.
The Poisson calculation includes the high-score tail and keeps full precision.

To make upcoming predictions while omitting current results entirely:

```powershell
python poisson_distribution.py --omit-current-season --output historical_only_predictions.csv
```

`--no-refresh-current` only disables the download; it still uses cached results.
`--omit-current-season` skips both downloading and reading current-season results.
The usual prediction command still writes its promotion audit JSON.

To replay another results file, supply `--results PATH --season YYYY-YY`; both
historical competitions must include the immediately preceding season.

Run the offline regression checks:

```powershell
python -m unittest test_backtest_poisson.py
```

This is a retrospective evaluation of the supplied data, not an independent audit
of its provenance or a prospective accuracy claim. Existing model parameters are
held fixed, including the five-game form window and promotion weighting. No
parameters were tuned against these outcomes. If they are tuned using this replay,
a later untouched period is needed for an independent test. A data-derived promotion
factor alone does not establish that the adjustment improves prediction quality.
