"""
verify_predictions.py — Compare model predictions vs actual results
=====================================================================
Shows expected games/sets vs what actually happened for each match.

Usage:
    python verify_predictions.py                                    # Miami Open
    python verify_predictions.py --file data/2026.csv               # full season
    python verify_predictions.py --file data/ongoing_tourneysxx.csv  # Miami only
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from build_model import Elo
from predict import predict, FEATURES, ELO


def run(file_path: str):
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded {len(df)} matches from {file_path}")

    # Need completed matches with scores
    df = df.dropna(subset=['winner_name', 'loser_name'])
    if 'w_svpt' in df.columns:
        df = df.dropna(subset=['w_svpt', 'l_svpt'])
        df = df[(df['w_svpt'] > 0) & (df['l_svpt'] > 0)]

    print(f"Completed matches with stats: {len(df)}")

    results = []
    np.random.seed(42)

    for i, (_, row) in enumerate(df.iterrows()):
        winner = row['winner_name']
        loser = row['loser_name']
        surface = row.get('surface', 'Hard')
        best_of = int(row['best_of']) if 'best_of' in row and not pd.isna(row.get('best_of')) else 3
        tourney = row.get('tourney_name', '')
        round_name = row.get('round', '')
        score = row.get('score', '')

        # Parse actual games from score
        actual_sets = []
        if isinstance(score, str) and score:
            for s in score.split():
                parts = s.replace('(', '-').replace(')', '').split('-')
                if len(parts) >= 2:
                    try:
                        g1, g2 = int(parts[0]), int(parts[1])
                        actual_sets.append((g1, g2))
                    except ValueError:
                        continue

        if not actual_sets:
            continue

        actual_total_games = sum(g1 + g2 for g1, g2 in actual_sets)
        actual_num_sets = len(actual_sets)
        actual_winner_games = sum(g1 for g1, g2 in actual_sets)
        actual_loser_games = sum(g2 for g1, g2 in actual_sets)
        actual_margin = actual_winner_games - actual_loser_games

        # Run prediction
        try:
            pred = predict(winner, loser, surface=surface, best_of=best_of, n_sims=5000)
        except Exception:
            continue
        if not pred:
            continue

        exp_games = pred['exp_games']
        exp_sets = pred['exp_sets']
        ensemble = pred['ensemble']
        win_prob = ensemble  # winner is always p1 here
        predicted_correct = ensemble >= 0.5

        games_error = exp_games - actual_total_games
        sets_error = exp_sets - actual_num_sets

        results.append({
            'tournament': tourney,
            'round': round_name,
            'surface': surface,
            'winner': winner,
            'loser': loser,
            'score': score,
            'actual_games': actual_total_games,
            'actual_sets': actual_num_sets,
            'actual_margin': actual_margin,
            'exp_games': round(exp_games, 1),
            'exp_sets': round(exp_sets, 2),
            'games_error': round(games_error, 1),
            'sets_error': round(sets_error, 2),
            'win_prob': round(win_prob, 3),
            'correct_winner': predicted_correct,
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(df)} processed...")

    if not results:
        print("No results generated.")
        return

    rdf = pd.DataFrame(results)

    # ── Print match-by-match ─────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  PREDICTION vs REALITY — {len(rdf)} matches")
    print(f"{'='*100}")
    print(f"  {'Winner':<22} {'Loser':<22} {'Score':<15} {'Act':>4} {'Exp':>5} {'Err':>5} {'Sets':>5} {'Prob':>6} {'✓':>3}")
    print(f"  {'─'*22} {'─'*22} {'─'*15} {'─'*4} {'─'*5} {'─'*5} {'─'*5} {'─'*6} {'─'*3}")

    for _, r in rdf.iterrows():
        check = "✅" if r['correct_winner'] else "❌"
        err_str = f"{r['games_error']:+.0f}"
        print(f"  {r['winner']:<22} {r['loser']:<22} {r['score']:<15} {r['actual_games']:>4} {r['exp_games']:>5} {err_str:>5} {r['actual_sets']:>3}/{r['exp_sets']:.1f} {r['win_prob']:>5.0%} {check:>3}")

    # ── Summary stats ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    correct = rdf['correct_winner'].sum()
    total = len(rdf)
    print(f"  Winner prediction: {correct}/{total} ({correct/total:.1%})")

    mae_games = rdf['games_error'].abs().mean()
    mae_sets = rdf['sets_error'].abs().mean()
    bias_games = rdf['games_error'].mean()
    bias_sets = rdf['sets_error'].mean()

    print(f"\n  Games prediction:")
    print(f"    Mean Absolute Error:  {mae_games:.1f} games")
    print(f"    Bias (+ = overpredict): {bias_games:+.1f} games")
    print(f"    Avg predicted: {rdf['exp_games'].mean():.1f}")
    print(f"    Avg actual:    {rdf['actual_games'].mean():.1f}")

    print(f"\n  Sets prediction:")
    print(f"    Mean Absolute Error:  {mae_sets:.2f} sets")
    print(f"    Bias (+ = overpredict): {bias_sets:+.2f} sets")
    print(f"    Avg predicted: {rdf['exp_sets'].mean():.2f}")
    print(f"    Avg actual:    {rdf['actual_sets'].mean():.2f}")

    # ── Games error distribution ─────────────────────────────────────
    print(f"\n  Games Error Distribution:")
    print(f"  {'Error':<15} {'Count':>6} {'Pct':>8}")
    print(f"  {'─'*15} {'─'*6} {'─'*8}")
    for lo, hi in [(-20, -5), (-5, -3), (-3, -1), (-1, 1), (1, 3), (3, 5), (5, 20)]:
        mask = (rdf['games_error'] >= lo) & (rdf['games_error'] < hi)
        n = mask.sum()
        if n == 0:
            continue
        label = f"{lo:+d} to {hi:+d}"
        print(f"  {label:<15} {n:>6} {n/total:>7.1%}")

    # ── Over/Under accuracy at common lines ──────────────────────────
    print(f"\n  Over/Under Accuracy at common lines:")
    print(f"  {'Line':<10} {'Over Pred':>10} {'Over Act':>10} {'Acc':>8}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    for line in [19.5, 20.5, 21.5, 22.5, 23.5]:
        pred_over = (rdf['exp_games'] > line).sum()
        actual_over = (rdf['actual_games'] > line).sum()
        model_says_over = rdf['exp_games'] > line
        actual_is_over = rdf['actual_games'] > line
        correct_ou = (model_says_over == actual_is_over).sum()
        print(f"  {line:<10} {pred_over:>10} {actual_over:>10} {correct_ou/total:>7.1%}")

    # ── By round ─────────────────────────────────────────────────────
    if 'round' in rdf.columns and rdf['round'].nunique() > 1:
        print(f"\n  By Round:")
        print(f"  {'Round':<10} {'Matches':>8} {'W Acc':>8} {'MAE Games':>10} {'Bias':>8}")
        print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
        for rnd in ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']:
            r_df = rdf[rdf['round'] == rnd]
            if len(r_df) == 0:
                continue
            w_acc = r_df['correct_winner'].mean()
            mae = r_df['games_error'].abs().mean()
            bias = r_df['games_error'].mean()
            print(f"  {rnd:<10} {len(r_df):>8} {w_acc:>7.1%} {mae:>9.1f} {bias:>+7.1f}")

    # ── Save ─────────────────────────────────────────────────────────
    out_path = BASE / "verify_results.csv"
    rdf.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify predictions vs actual results")
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    if args.file:
        file_path = args.file
    else:
        # Default: try ongoing_tourneysxx (Miami)
        for name in ['ongoing_tourneysxx.csv', 'ongoing_tourneys_backup.csv', 'ongoing_tourneys.csv']:
            p = BASE / "data" / name
            if p.exists():
                file_path = str(p)
                break
        else:
            file_path = str(BASE / "data" / "2026.csv")

    run(file_path)
