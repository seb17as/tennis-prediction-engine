"""
backtest.py — Test model accuracy on completed matches
========================================================
Runs your predict() on matches that already have results,
compares predictions vs actual winners.

Usage:
    python backtest.py                          # backtest ongoing_tourneys
    python backtest.py --file data/2025.csv     # backtest a full season
    python backtest.py --file data/2026.csv --min-prob 0.60  # filter by confidence
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from build_model import Elo
from predict import predict, FEATURES, ELO

def run_backtest(file_path: str, min_prob: float = 0.0, surface_filter: str = None):
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded {len(df)} matches from {file_path}")
    print(f"Columns: {list(df.columns)[:10]}...")

    # Need winner_name and loser_name
    if 'winner_name' not in df.columns or 'loser_name' not in df.columns:
        print("ERROR: File needs 'winner_name' and 'loser_name' columns")
        return

    # Filter to completed matches with stats
    if 'w_svpt' in df.columns:
        df = df.dropna(subset=['w_svpt', 'l_svpt'])
        df = df[(df['w_svpt'] > 0) & (df['l_svpt'] > 0)]

    if surface_filter:
        df = df[df['surface'] == surface_filter]

    print(f"Testing on {len(df)} completed matches")
    if surface_filter:
        print(f"Surface filter: {surface_filter}")

    # ── Run predictions ──────────────────────────────────────────────────
    results = []
    skipped = 0

    for i, (_, row) in enumerate(df.iterrows()):
        winner = row['winner_name']
        loser = row['loser_name']
        surface = row.get('surface', 'Hard')
        best_of = int(row['best_of']) if 'best_of' in row and not pd.isna(row.get('best_of')) else 3
        tourney = row.get('tourney_name', '')
        round_name = row.get('round', '')

        # Randomize who is player_1 to avoid bias
        if np.random.random() > 0.5:
            p1, p2 = winner, loser
            actual_winner_is_p1 = True
        else:
            p1, p2 = loser, winner
            actual_winner_is_p1 = False

        try:
            pred = predict(p1, p2, surface=surface, best_of=best_of, n_sims=5000)
        except Exception:
            skipped += 1
            continue

        if not pred:
            skipped += 1
            continue

        ensemble = pred['ensemble']
        predicted_p1_wins = ensemble >= 0.5
        win_prob = ensemble if predicted_p1_wins else (1 - ensemble)

        # Apply confidence filter
        if win_prob < min_prob:
            continue

        # Did we predict correctly?
        if predicted_p1_wins:
            predicted_winner = p1
        else:
            predicted_winner = p2

        correct = predicted_winner == winner

        results.append({
            'tournament': tourney,
            'round': round_name,
            'surface': surface,
            'p1': p1,
            'p2': p2,
            'actual_winner': winner,
            'predicted_winner': predicted_winner,
            'win_prob': round(win_prob, 4),
            'elo_prob': round(pred['elo_prob'] if predicted_p1_wins else 1 - pred['elo_prob'], 4),
            'mc_prob': round(pred['mc_prob'] if predicted_p1_wins else 1 - pred['mc_prob'], 4),
            'analytical_prob': round(pred['analytical'] if predicted_p1_wins else 1 - pred['analytical'], 4),
            'correct': correct,
        })

        if (i + 1) % 50 == 0:
            acc_so_far = sum(r['correct'] for r in results) / len(results) if results else 0
            print(f"  {i+1}/{len(df)} processed... ({len(results)} predicted, {acc_so_far:.1%} accuracy)")

    if not results:
        print("\nNo predictions generated. Check player names in the file.")
        return

    # ── Results ──────────────────────────────────────────────────────────
    rdf = pd.DataFrame(results)
    total = len(rdf)
    correct = rdf['correct'].sum()
    accuracy = correct / total

    print(f"\n{'='*65}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*65}")
    print(f"  Matches predicted:  {total}")
    print(f"  Skipped:            {skipped}")
    print(f"  Correct:            {correct}")
    print(f"  Accuracy:           {accuracy:.1%}")

    # ── By confidence bucket ─────────────────────────────────────────────
    print(f"\n  {'Confidence':<20} {'Total':>8} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*10}")

    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
               (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 1.01)]
    for lo, hi in buckets:
        mask = (rdf['win_prob'] >= lo) & (rdf['win_prob'] < hi)
        bucket_df = rdf[mask]
        if len(bucket_df) == 0:
            continue
        b_correct = bucket_df['correct'].sum()
        b_total = len(bucket_df)
        b_acc = b_correct / b_total
        label = f"{lo:.0%} - {hi:.0%}" if hi < 1.0 else f"{lo:.0%}+"
        print(f"  {label:<20} {b_total:>8} {b_correct:>8} {b_acc:>9.1%}")

    # ── By surface ───────────────────────────────────────────────────────
    print(f"\n  {'Surface':<20} {'Total':>8} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*10}")
    for surf in rdf['surface'].unique():
        s_df = rdf[rdf['surface'] == surf]
        s_correct = s_df['correct'].sum()
        s_total = len(s_df)
        s_acc = s_correct / s_total
        print(f"  {surf:<20} {s_total:>8} {s_correct:>8} {s_acc:>9.1%}")

    # ── By model component ───────────────────────────────────────────────
    print(f"\n  Model Component Accuracy:")
    print(f"  {'Model':<20} {'Accuracy':>10}")
    print(f"  {'─'*20} {'─'*10}")

    elo_correct = (rdf['elo_prob'] > 0.5).eq(rdf['correct']).sum()
    mc_correct = (rdf['mc_prob'] > 0.5).eq(rdf['correct']).sum()
    ana_correct = (rdf['analytical_prob'] > 0.5).eq(rdf['correct']).sum()

    # Simpler: just check if each sub-model's favored player won
    elo_acc = sum(1 for _, r in rdf.iterrows() if (r['elo_prob'] > 0.5) == r['correct']) / total
    mc_acc = sum(1 for _, r in rdf.iterrows() if (r['mc_prob'] > 0.5) == r['correct']) / total
    ana_acc = sum(1 for _, r in rdf.iterrows() if (r['analytical_prob'] > 0.5) == r['correct']) / total

    print(f"  {'Elo':<20} {elo_acc:>9.1%}")
    print(f"  {'Analytical':<20} {ana_acc:>9.1%}")
    print(f"  {'Monte Carlo':<20} {mc_acc:>9.1%}")
    print(f"  {'Ensemble':<20} {accuracy:>9.1%}")

    # ── Calibration check ────────────────────────────────────────────────
    print(f"\n  Calibration (predicted vs actual):")
    print(f"  {'Predicted':<20} {'Actual Win%':>12} {'N':>8}")
    print(f"  {'─'*20} {'─'*12} {'─'*8}")
    for lo, hi in [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]:
        mask = (rdf['win_prob'] >= lo) & (rdf['win_prob'] < hi)
        bucket_df = rdf[mask]
        if len(bucket_df) == 0:
            continue
        actual_win = bucket_df['correct'].mean()
        label = f"{lo:.0%}-{hi:.0%}" if hi < 1.0 else f"{lo:.0%}+"
        print(f"  {label:<20} {actual_win:>11.1%} {len(bucket_df):>8}")

    # ── Save detailed results ────────────────────────────────────────────
    out_path = BASE / "backtest_results.csv"
    rdf.to_csv(out_path, index=False)
    print(f"\n  Detailed results saved → {out_path}")


if __name__ == "__main__":
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Backtest tennis predictions")
    parser.add_argument("--file", type=str, default=None,
                        help="CSV file to backtest (default: ongoing_tourneys_backup.csv)")
    parser.add_argument("--min-prob", type=float, default=0.0,
                        help="Only count picks above this confidence threshold")
    parser.add_argument("--surface", type=str, default=None,
                        help="Filter by surface: Hard, Clay, Grass")
    args = parser.parse_args()

    if args.file:
        file_path = args.file
    else:
        # Try ongoing_tourneys_backup first, then 2026.csv
        backup = BASE / "data" / "ongoing_tourneys_backup.csv"
        if backup.exists():
            file_path = str(backup)
        else:
            file_path = str(BASE / "data" / "2026.csv")

    run_backtest(file_path, min_prob=args.min_prob, surface_filter=args.surface)
