"""Generate interactive, standalone matchday dashboard HTML for GitHub Pages."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import team_name
from .predictor import Predictor


def build_dashboard(output_path: Path | str = 'docs/index.html',
                    fixtures_path: Path | str = 'upcoming_premier_league_fixtures.csv',
                    artifact_dir: Path | str = 'artifacts/hybrid_v1',
                    players_path: Path | str = 'players.csv',
                    appearances_path: Path | str = 'fotmob_player_appearances.csv',
                    results_path: Path | str = 'current_premier_league_results.csv',
                    max_fixtures: int = 10) -> Path:
    """Generate interactive dashboard HTML with client-side Poisson recalculation."""
    fixtures_file = Path(fixtures_path)
    if not fixtures_file.exists():
        raise FileNotFoundError(f"Fixtures file not found: {fixtures_file}")

    all_fixtures = pd.read_csv(fixtures_file)
    for col in ['home_team', 'away_team']:
        all_fixtures[col] = all_fixtures[col].map(team_name)

    upcoming = all_fixtures.head(max_fixtures).copy()

    model = Predictor(artifact_dir)
    ea = pd.read_csv(players_path) if Path(players_path).exists() else None
    apps = pd.read_csv(appearances_path) if Path(appearances_path).exists() else None

    predictions = model.predict(upcoming, ea, apps, filter_future=True)

    # Compute recent 5-match W/D/L form from results
    club_recent_form = {}
    if Path(results_path).exists():
        results = pd.read_csv(results_path)
        results['home_team'] = results.home_team.map(team_name)
        results['away_team'] = results.away_team.map(team_name)
        clubs = set(upcoming.home_team.unique()).union(upcoming.away_team.unique())
        for club in clubs:
            c_matches = results[(results.home_team == club) | (results.away_team == club)].sort_values('date').tail(5)
            seq = []
            for _, m in c_matches.iterrows():
                if m.home_team == club:
                    if m.home_goals > m.away_goals: seq.append('W')
                    elif m.home_goals == m.away_goals: seq.append('D')
                    else: seq.append('L')
                else:
                    if m.away_goals > m.home_goals: seq.append('W')
                    elif m.away_goals == m.home_goals: seq.append('D')
                    else: seq.append('L')
            club_recent_form[club] = seq

    matches_payload = []
    for p in predictions:
        ht = p['home_team']
        at = p['away_team']
        match_entry = {
            'fixture_id': p['fixture_id'],
            'home_team': ht,
            'away_team': at,
            'kickoff_utc': p['kickoff_utc'],
            'home_win_prob': round(p['home_win_probability'], 4),
            'draw_prob': round(p['draw_probability'], 4),
            'away_win_prob': round(p['away_win_probability'], 4),
            'exp_home_goals': round(p['expected_home_goals'], 3),
            'exp_away_goals': round(p['expected_away_goals'], 3),
            'most_likely_score': p['most_likely_score'],
            'top_scores': p['top_five_scorelines'],
            'btts': round(p['both_teams_to_score'], 3),
            'over_2_5': round(p['over_2_5'], 3),
            'under_2_5': round(p['under_2_5'], 3),
            'home_recent': club_recent_form.get(ht, []),
            'away_recent': club_recent_form.get(at, []),
        }
        matches_payload.append(match_entry)

    matches_json = json.dumps(matches_payload)
    rho_val = float(model.dc.rho)
    beta_val = 0.1675
    generated_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Premier League Matchday Predictor</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 9999px; }}
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased min-h-screen p-4 md:p-6 transition-colors">
  <div class="max-w-6xl mx-auto space-y-6">

    <!-- Header Section -->
    <header class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[var(--border)] pb-5">
      <div>
        <div class="flex items-center gap-2 mb-1">
          <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
            Live EPL Forecasts
          </span>
          <span class="text-xs text-[var(--muted-foreground)]">Model: Dixon-Coles + Form Scaler</span>
        </div>
        <h1 class="text-2xl md:text-3xl font-bold tracking-tight">Premier League Match Predictor</h1>
        <p class="text-sm text-[var(--muted-foreground)] mt-0.5">
          Prospectively validated Poisson goal rates with live FotMob starter form simulation.
        </p>
      </div>
      <div class="flex items-center gap-2 flex-wrap text-xs">
        <div class="bg-[var(--card)] border border-[var(--border)] rounded-lg px-3 py-2 shadow-sm text-center">
          <span class="text-[var(--muted-foreground)] block">Last Updated</span>
          <span class="font-semibold text-emerald-500">{generated_time}</span>
        </div>
        <div class="bg-[var(--card)] border border-[var(--border)] rounded-lg px-3 py-2 shadow-sm text-center">
          <span class="text-[var(--muted-foreground)] block">Model Cutoff Fit</span>
          <span class="font-semibold text-cyan-500">Log Loss 0.9619</span>
        </div>
        <div class="bg-[var(--card)] border border-[var(--border)] rounded-lg px-3 py-2 shadow-sm text-center">
          <span class="text-[var(--muted-foreground)] block">Form Scaling &beta;</span>
          <span class="font-semibold text-indigo-500">{beta_val}</span>
        </div>
      </div>
    </header>

    <!-- Main Layout: Fixture List & Deep Dive -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

      <!-- Fixtures Column (5 cols) -->
      <div class="lg:col-span-5 space-y-3">
        <div class="flex items-center justify-between">
          <h2 class="text-base font-semibold flex items-center gap-2">
            <span>Upcoming Fixtures</span>
            <span id="fixture-count" class="text-xs bg-[var(--content)] px-2 py-0.5 rounded-full border border-[var(--border)] text-[var(--muted-foreground)]">{len(matches_payload)}</span>
          </h2>
          <span class="text-xs text-[var(--muted-foreground)]">Click to inspect</span>
        </div>

        <div id="fixtures-list" class="space-y-2 max-h-[750px] overflow-y-auto pr-1"></div>
      </div>

      <!-- Match Inspector Column (7 cols) -->
      <div class="lg:col-span-7">
        <div id="inspector-card" class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 md:p-6 shadow-sm sticky top-6 space-y-6"></div>
      </div>

    </div>

    <!-- Methodology & Mathematical Guarantees Footer -->
    <footer class="mt-8 pt-5 border-t border-[var(--border)] text-xs text-[var(--muted-foreground)] flex flex-col md:flex-row justify-between items-center gap-3">
      <div>
        <span>Dixon & Coles (1997) maximum likelihood with time decay (&tau; = 180d) and low-score correlation (&rho; = {rho_val:.4f}).</span>
      </div>
      <div class="flex items-center gap-4">
        <span>No Lookahead Leakage</span>
        <span>•</span>
        <span>Strict Next-UTC-Day Availability</span>
        <span>•</span>
        <span>Reconciled Coherent Markets</span>
      </div>
    </footer>

  </div>

  <script>
    const MATCHES = {matches_json};
    const RHO = {rho_val};
    const DEFAULT_BETA = {beta_val};

    let selectedIndex = 0;
    let customDelta = 0.0;

    function poissonPmf(k, lambda) {{
      if (k < 0 || lambda <= 0) return 0;
      let logP = -lambda + k * Math.log(lambda);
      for (let i = 2; i <= k; i++) logP -= Math.log(i);
      return Math.exp(logP);
    }}

    function tau(x, y, lam, mu, rho) {{
      if (x === 0 && y === 0) return 1.0 - lam * mu * rho;
      if (x === 0 && y === 1) return 1.0 + lam * rho;
      if (x === 1 && y === 0) return 1.0 + mu * rho;
      if (x === 1 && y === 1) return 1.0 - rho;
      return 1.0;
    }}

    function computeMarkets(lam, mu, rho) {{
      const MAX_GOALS = 6;
      let matrix = [];
      let totalMass = 0.0;

      for (let h = 0; h <= MAX_GOALS; h++) {{
        matrix[h] = [];
        for (let a = 0; a <= MAX_GOALS; a++) {{
          let t = tau(h, a, lam, mu, rho);
          if (t <= 0) t = 1e-6;
          let p = poissonPmf(h, lam) * poissonPmf(a, mu) * t;
          matrix[h][a] = p;
          totalMass += p;
        }}
      }}

      let pHome = 0.0, pDraw = 0.0, pAway = 0.0;
      let btts = 0.0, over25 = 0.0;
      let scores = [];

      for (let h = 0; h <= MAX_GOALS; h++) {{
        for (let a = 0; a <= MAX_GOALS; a++) {{
          matrix[h][a] /= totalMass;
          let prob = matrix[h][a];
          if (h > a) pHome += prob;
          else if (h === a) pDraw += prob;
          else pAway += prob;

          if (h > 0 && a > 0) btts += prob;
          if (h + a > 2.5) over25 += prob;

          scores.push({{ score: `${{h}}-${{a}}`, prob: prob }});
        }}
      }}

      scores.sort((a, b) => b.prob - a.prob);

      return {{
        pHome, pDraw, pAway,
        btts, over25, under25: 1.0 - over25,
        mostLikelyScore: scores[0].score,
        topScores: scores.slice(0, 5)
      }};
    }}

    function formatDate(isoStr) {{
      const d = new Date(isoStr);
      return d.toLocaleDateString('en-GB', {{ weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', timeZone: 'UTC' }}) + ' UTC';
    }}

    function renderFixtureList() {{
      const listEl = document.getElementById('fixtures-list');
      listEl.innerHTML = '';

      MATCHES.forEach((m, idx) => {{
        const isSelected = idx === selectedIndex;
        const card = document.createElement('div');
        card.className = `p-3.5 rounded-xl border cursor-pointer transition-all ${{
          isSelected
            ? 'bg-[var(--card)] border-emerald-500 ring-2 ring-emerald-500/20 shadow-md'
            : 'bg-[var(--card)] border-[var(--border)] hover:border-emerald-500/50 hover:bg-[var(--content)] shadow-sm'
        }}`;
        card.onclick = () => {{
          selectedIndex = idx;
          customDelta = 0.0;
          renderFixtureList();
          renderInspector();
        }};

        const pHomePct = Math.round(m.home_win_prob * 100);
        const pDrawPct = Math.round(m.draw_prob * 100);
        const pAwayPct = Math.round(m.away_win_prob * 100);

        card.innerHTML = `
          <div class="flex items-center justify-between text-xs text-[var(--muted-foreground)] mb-1.5">
            <span>${{formatDate(m.kickoff_utc)}}</span>
            <span class="font-mono bg-[var(--content)] px-1.5 py-0.5 rounded border border-[var(--border)] text-[var(--foreground)] font-medium">${{m.most_likely_score}}</span>
          </div>
          <div class="flex items-center justify-between font-semibold text-sm mb-2">
            <span class="truncate pr-1">${{m.home_team}}</span>
            <span class="text-xs text-[var(--muted-foreground)] px-1">vs</span>
            <span class="truncate pl-1 text-right">${{m.away_team}}</span>
          </div>
          <div class="w-full bg-[var(--content)] h-2 rounded-full overflow-hidden flex border border-[var(--border)]">
            <div style="width: ${{pHomePct}}%" class="bg-emerald-500" title="Home: ${{pHomePct}}%"></div>
            <div style="width: ${{pDrawPct}}%" class="bg-slate-400 dark:bg-slate-500" title="Draw: ${{pDrawPct}}%"></div>
            <div style="width: ${{pAwayPct}}%" class="bg-indigo-500" title="Away: ${{pAwayPct}}%"></div>
          </div>
          <div class="flex justify-between text-[11px] font-mono text-[var(--muted-foreground)] mt-1">
            <span class="text-emerald-500 font-semibold">${{pHomePct}}%</span>
            <span>${{pDrawPct}}%</span>
            <span class="text-indigo-500 font-semibold">${{pAwayPct}}%</span>
          </div>
        `;
        listEl.appendChild(card);
      }});
    }}

    function renderRecentBadges(arr) {{
      if (!arr || !arr.length) return '<span class="text-xs text-[var(--muted-foreground)]">N/A</span>';
      return arr.map(res => {{
        const color = res === 'W' ? 'bg-emerald-500/20 text-emerald-500 border-emerald-500/30' :
                      (res === 'D' ? 'bg-amber-500/20 text-amber-500 border-amber-500/30' : 'bg-rose-500/20 text-rose-500 border-rose-500/30');
        return `<span class="inline-flex items-center justify-center w-5 h-5 text-[11px] font-bold rounded border ${{color}}">${{res}}</span>`;
      }}).join(' ');
    }}

    function renderInspector() {{
      const m = MATCHES[selectedIndex];
      const cardEl = document.getElementById('inspector-card');

      const expHomeSim = Math.max(0.05, Math.min(10.0, m.exp_home_goals * Math.exp(DEFAULT_BETA * customDelta)));
      const expAwaySim = Math.max(0.05, Math.min(10.0, m.exp_away_goals * Math.exp(-DEFAULT_BETA * customDelta)));
      const sim = computeMarkets(expHomeSim, expAwaySim, RHO);

      const pH_base = Math.round(m.home_win_prob * 100);
      const pD_base = Math.round(m.draw_prob * 100);
      const pA_base = Math.round(m.away_win_prob * 100);

      const pH_sim = Math.round(sim.pHome * 100);
      const pD_sim = Math.round(sim.pDraw * 100);
      const pA_sim = Math.round(sim.pAway * 100);

      const deltaH = pH_sim - pH_base;
      const deltaA = pA_sim - pA_base;

      cardEl.innerHTML = `
        <div class="border-b border-[var(--border)] pb-4">
          <div class="flex items-center justify-between text-xs text-[var(--muted-foreground)] mb-2">
            <span>Premier League Forecast</span>
            <span>${{formatDate(m.kickoff_utc)}}</span>
          </div>
          <div class="grid grid-cols-5 items-center gap-2">
            <div class="col-span-2 text-left">
              <h3 class="text-xl md:text-2xl font-bold truncate">${{m.home_team}}</h3>
              <div class="mt-1 flex items-center gap-1">${{renderRecentBadges(m.home_recent)}}</div>
            </div>
            <div class="col-span-1 text-center font-mono font-bold text-lg text-[var(--muted-foreground)]">
              vs
            </div>
            <div class="col-span-2 text-right">
              <h3 class="text-xl md:text-2xl font-bold truncate">${{m.away_team}}</h3>
              <div class="mt-1 flex items-center justify-end gap-1">${{renderRecentBadges(m.away_recent)}}</div>
            </div>
          </div>
        </div>

        <div class="bg-[var(--content)] border border-[var(--border)] rounded-xl p-4 space-y-3">
          <div class="flex items-center justify-between">
            <span class="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">Win / Draw / Loss Probabilities</span>
            ${{customDelta !== 0 ? `
              <span class="text-xs px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-500 font-medium border border-indigo-500/20">
                Form Adjusted (&Delta; ${{customDelta > 0 ? '+' : ''}}${{customDelta.toFixed(2)}})
              </span>
            ` : '<span class="text-xs text-[var(--muted-foreground)]">Pure Dixon-Coles Baseline</span>'}}
          </div>

          <div class="grid grid-cols-3 gap-2 text-center">
            <div class="p-3 bg-[var(--card)] rounded-lg border border-[var(--border)]">
              <span class="text-xs text-[var(--muted-foreground)] block">${{m.home_team}}</span>
              <span class="text-2xl font-black text-emerald-500">${{pH_sim}}%</span>
              ${{customDelta !== 0 ? `<span class="text-[11px] block ${{deltaH >= 0 ? 'text-emerald-500' : 'text-rose-500'}}">${{deltaH >= 0 ? '+' : ''}}${{deltaH}}% vs base</span>` : ''}}
            </div>
            <div class="p-3 bg-[var(--card)] rounded-lg border border-[var(--border)]">
              <span class="text-xs text-[var(--muted-foreground)] block">Draw</span>
              <span class="text-2xl font-black text-slate-500 dark:text-slate-400">${{pD_sim}}%</span>
              ${{customDelta !== 0 ? `<span class="text-[11px] block text-[var(--muted-foreground)]">${{pD_sim - pD_base >= 0 ? '+' : ''}}${{pD_sim - pD_base}}%</span>` : ''}}
            </div>
            <div class="p-3 bg-[var(--card)] rounded-lg border border-[var(--border)]">
              <span class="text-xs text-[var(--muted-foreground)] block">${{m.away_team}}</span>
              <span class="text-2xl font-black text-indigo-500">${{pA_sim}}%</span>
              ${{customDelta !== 0 ? `<span class="text-[11px] block ${{deltaA >= 0 ? 'text-indigo-500' : 'text-rose-500'}}">${{deltaA >= 0 ? '+' : ''}}${{deltaA}}% vs base</span>` : ''}}
            </div>
          </div>

          <div class="w-full bg-[var(--card)] h-3 rounded-full overflow-hidden flex border border-[var(--border)] mt-2">
            <div style="width: ${{pH_sim}}%" class="bg-emerald-500 transition-all duration-300"></div>
            <div style="width: ${{pD_sim}}%" class="bg-slate-400 dark:bg-slate-500 transition-all duration-300"></div>
            <div style="width: ${{pA_sim}}%" class="bg-indigo-500 transition-all duration-300"></div>
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="p-4 rounded-xl border border-[var(--border)] bg-[var(--content)] space-y-3">
            <span class="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] block">Goal Expectancy</span>
            <div class="flex items-center justify-between text-sm py-1 border-b border-[var(--border)]">
              <span>${{m.home_team}} Exp Goals</span>
              <span class="font-mono font-bold text-emerald-500">${{expHomeSim.toFixed(2)}}</span>
            </div>
            <div class="flex items-center justify-between text-sm py-1 border-b border-[var(--border)]">
              <span>${{m.away_team}} Exp Goals</span>
              <span class="font-mono font-bold text-indigo-500">${{expAwaySim.toFixed(2)}}</span>
            </div>
            <div class="flex items-center justify-between text-sm py-1 border-b border-[var(--border)]">
              <span>Both Teams To Score (BTTS)</span>
              <span class="font-mono font-bold">${{Math.round(sim.btts * 100)}}%</span>
            </div>
            <div class="flex items-center justify-between text-sm py-1">
              <span>Over / Under 2.5 Goals</span>
              <span class="font-mono font-bold">${{Math.round(sim.over25 * 100)}}% / ${{Math.round(sim.under25 * 100)}}%</span>
            </div>
          </div>

          <div class="p-4 rounded-xl border border-[var(--border)] bg-[var(--content)] space-y-3">
            <span class="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] block">Top Scorelines</span>
            <div class="space-y-1.5">
              ${{sim.topScores.map((s, idx) => `
                <div class="flex items-center justify-between text-xs">
                  <span class="font-mono font-bold w-10">${{s.score}}</span>
                  <div class="flex-1 mx-3 bg-[var(--card)] h-2 rounded-full overflow-hidden border border-[var(--border)]">
                    <div style="width: ${{Math.round(s.prob * 500)}}%" class="bg-cyan-500 h-full rounded-full"></div>
                  </div>
                  <span class="font-mono text-[var(--muted-foreground)] w-10 text-right">${{(s.prob * 100).toFixed(1)}}%</span>
                </div>
              `).join('')}}
            </div>
          </div>
        </div>

        <div class="p-4 rounded-xl border border-indigo-500/30 bg-indigo-500/5 space-y-3">
          <div class="flex items-center justify-between">
            <div>
              <h4 class="text-sm font-semibold text-indigo-400">Live Starter Form Simulator</h4>
              <p class="text-xs text-[var(--muted-foreground)]">Simulate confirmed lineup sharpness difference (&beta; = {beta_val})</p>
            </div>
            <span class="font-mono text-sm font-bold px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
              ${{customDelta > 0 ? '+' : ''}}${{customDelta.toFixed(2)}}
            </span>
          </div>

          <div class="space-y-1">
            <input type="range" min="-2.0" max="2.0" step="0.05" value="${{customDelta}}"
              id="delta-slider"
              class="w-full accent-indigo-500 cursor-pointer"
              oninput="onSliderChange(this.value)">
            <div class="flex justify-between text-[11px] text-[var(--muted-foreground)] font-mono">
              <span>Away Heavy Form (-2.0)</span>
              <span>Neutral (0.0)</span>
              <span>Home Heavy Form (+2.0)</span>
            </div>
          </div>

          <div class="flex items-center gap-2 pt-1">
            <button onclick="setPreset(-1.0)" class="text-xs px-2.5 py-1 rounded bg-[var(--card)] border border-[var(--border)] hover:border-indigo-500 hover:text-indigo-400 transition-colors">
              Away Edge (-1.0)
            </button>
            <button onclick="setPreset(0.0)" class="text-xs px-2.5 py-1 rounded bg-[var(--card)] border border-[var(--border)] hover:border-indigo-500 hover:text-indigo-400 transition-colors">
              Reset (0.0)
            </button>
            <button onclick="setPreset(1.0)" class="text-xs px-2.5 py-1 rounded bg-[var(--card)] border border-[var(--border)] hover:border-indigo-500 hover:text-indigo-400 transition-colors">
              Home Edge (+1.0)
            </button>
          </div>
        </div>
      `;
    }}

    function onSliderChange(val) {{
      customDelta = parseFloat(val);
      renderInspector();
    }}

    function setPreset(val) {{
      customDelta = val;
      renderInspector();
    }}

    renderFixtureList();
    renderInspector();
  </script>
</body>
</html>
'''

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html_content, encoding='utf-8')
    return out_file
