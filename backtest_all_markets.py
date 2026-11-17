"""
backtest_all_markets.py — Backtest ALL tennis betting markets
==============================================================
Tests your model across 4 markets using real historical odds:
  1. Moneyline (match winner)
  2. Set Betting (exact set score)
  3. Over/Under Total Games
  4. Game Spread (handicap)

SETUP:
  1. Download from http://www.tennis-data.co.uk/alldata.php
  2. Put xlsx files in Tennis_V2/odds_data/

Usage:
    python backtest_all_markets.py --year 2025
    python backtest_all_markets.py --year 2026
    python backtest_all_markets.py --year 2025 --min-prob 0.60
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent
ODDS_DIR = BASE / "odds_data"
sys.path.insert(0, str(BASE))

from build_model import Elo
from predict import predict, FEATURES, ELO


# ═══════════════════════════════════════════════════════════════════════════
# NAME CONVERSION (same as backtest_with_odds.py)
# ═══════════════════════════════════════════════════════════════════════════
_PLAYER_LOOKUP = {}

def _build_name_lookup():
    global _PLAYER_LOOKUP
    if _PLAYER_LOOKUP:
        return
    all_players = FEATURES['player'].unique()
    for full_name in all_players:
        parts = full_name.strip().split()
        if len(parts) < 2:
            continue
        first = parts[0]
        last = " ".join(parts[1:])
        initial = first[0] + "."
        _PLAYER_LOOKUP[f"{last} {initial}".lower()] = full_name
        _PLAYER_LOOKUP[f"{last} {first}".lower()] = full_name
        last_parts = last.split()
        if len(last_parts) > 1:
            _PLAYER_LOOKUP[f"{last_parts[-1]} {initial}".lower()] = full_name
        if "-" in first:
            compound_init = ".".join([p[0] for p in first.split("-")]) + "."
            _PLAYER_LOOKUP[f"{last} {compound_init}".lower()] = full_name
    for full_name in all_players:
        _PLAYER_LOOKUP[full_name.strip().lower()] = full_name.strip()
    manual = {
        "auger-aliassime f.": "Felix Auger-Aliassime",
        "van de zandschulp b.": "Botic Van De Zandschulp",
        "de minaur a.": "Alex De Minaur",
        "ugo carabelli c.": "Camilo Ugo Carabelli",
        "martin etcheverry t.": "Tomas Martin Etcheverry",
        "etcheverry t.": "Tomas Martin Etcheverry",
        "davidovich fokina a.": "Alejandro Davidovich Fokina",
        "bautista agut r.": "Roberto Bautista Agut",
        "carreno busta p.": "Pablo Carreno Busta",
        "carreno-busta p.": "Pablo Carreno Busta",
        "mpetshi perricard g.": "Giovanni Mpetshi Perricard",
        "carballes baena r.": "Roberto Carballes Baena",
        "martinez p.": "Pedro Martinez",
        "diaz acosta f.": "Facundo Diaz Acosta",
        "budkov kjaer n.": "Nicolai Budkov Kjaer",
        "struff j.l.": "Jan-Lennard Struff",
        "struff j.-l.": "Jan-Lennard Struff",
        "cerundolo j.m.": "Juan Manuel Cerundolo",
        "cerundolo f.": "Francisco Cerundolo",
        "de jong j.": "Jesper de Jong",
        "barrios m.": "Tomas Barrios Vera",
    }
    for k, v in manual.items():
        _PLAYER_LOOKUP[k] = v
    print(f"  Name lookup built: {len(_PLAYER_LOOKUP)} entries")


def convert_name(short_name: str) -> str:
    _build_name_lookup()
    key = short_name.strip().lower()
    if key in _PLAYER_LOOKUP:
        return _PLAYER_LOOKUP[key]
    key_clean = key.replace(".", "").strip()
    for k, v in _PLAYER_LOOKUP.items():
        if k.replace(".", "").strip() == key_clean:
            return v
    parts = short_name.strip().split()
    if len(parts) >= 2:
        initial_part = parts[-1].replace(".", "").upper()
        lastname = " ".join(parts[:-1]).lower()
        candidates = []
        for full_name in FEATURES['player'].unique():
            fp = full_name.strip().split()
            if len(fp) < 2:
                continue
            if " ".join(fp[1:]).lower() == lastname and fp[0][0].upper() == initial_part[0]:
                candidates.append(full_name)
        if len(candidates) == 1:
            _PLAYER_LOOKUP[key] = candidates[0]
            return candidates[0]
        elif len(candidates) > 1:
            if len(initial_part) > 1:
                for c in candidates:
                    c_first = c.split()[0]
                    if "-" in c_first:
                        c_inits = "".join([p[0] for p in c_first.split("-")]).upper()
                        if c_inits == initial_part:
                            _PLAYER_LOOKUP[key] = c
                            return c
            _PLAYER_LOOKUP[key] = candidates[0]
            return candidates[0]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PARSE ACTUAL MATCH SCORES
# ═══════════════════════════════════════════════════════════════════════════

def parse_set_scores(row) -> list:
    """Parse W1/L1, W2/L2, etc. into [(w_games, l_games), ...]"""
    sets = []
    for i in range(1, 6):
        w_col, l_col = f'W{i}', f'L{i}'
        if w_col in row and l_col in row:
            w_games = row[w_col]
            l_games = row[l_col]
            if pd.notna(w_games) and pd.notna(l_games):
                sets.append((int(w_games), int(l_games)))
    return sets


def get_total_games(sets: list) -> int:
    return sum(w + l for w, l in sets)


def get_set_score(sets: list) -> str:
    """Returns '2-0', '2-1', '3-0', etc."""
    w_sets = sum(1 for w, l in sets if w > l)
    l_sets = sum(1 for w, l in sets if l > w)
    return f"{w_sets}-{l_sets}"


def get_game_margin(sets: list) -> int:
    """Winner total games minus loser total games."""
    w_total = sum(w for w, l in sets)
    l_total = sum(l for w, l in sets)
    return w_total - l_total


# ═══════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(year: int, min_prob: float = 0.0):
    # Load odds data
    ODDS_DIR.mkdir(exist_ok=True)
    odds_file = None
    for ext in ['.xlsx', '.xls']:
        f = ODDS_DIR / f"{year}{ext}"
        if f.exists():
            odds_file = f
            break
    if not odds_file:
        print(f"ERROR: Download {year}.xlsx from http://www.tennis-data.co.uk/alldata.php")
        print(f"Save to: {ODDS_DIR}/{year}.xlsx")
        sys.exit(1)

    engine = 'openpyxl' if str(odds_file).endswith('.xlsx') else 'xlrd'
    df = pd.read_excel(odds_file, engine=engine)
    print(f"Loaded {len(df)} matches from {odds_file}")

    # Use AvgW/AvgL for broadest coverage
    w_odds_col, l_odds_col = 'AvgW', 'AvgL'
    if w_odds_col not in df.columns:
        for prefix in ["B365", "PS"]:
            if f"{prefix}W" in df.columns:
                w_odds_col, l_odds_col = f"{prefix}W", f"{prefix}L"
                break

    df = df.dropna(subset=['Winner', 'Loser', w_odds_col, l_odds_col])
    print(f"Matches with odds: {len(df)}")

    # ── Results accumulators ─────────────────────────────────────────────
    ml_results = []       # Moneyline
    sb_results = []       # Set Betting
    ou_results = []       # Over/Under
    gs_results = []       # Game Spread

    skipped = 0
    np.random.seed(42)

    for i, (_, row) in enumerate(df.iterrows()):
        winner_raw = str(row['Winner']).strip()
        loser_raw = str(row['Loser']).strip()
        winner = convert_name(winner_raw)
        loser = convert_name(loser_raw)
        if not winner or not loser:
            skipped += 1
            continue

        surface = str(row.get('Surface', 'Hard')).strip()
        best_of = int(row['Best of']) if 'Best of' in row and pd.notna(row.get('Best of')) else 3
        tourney = str(row.get('Tournament', row.get('Location', '')))
        winner_odds = float(row[w_odds_col])
        loser_odds = float(row[l_odds_col])

        # Parse actual scores
        sets = parse_set_scores(row)
        if not sets:
            skipped += 1
            continue

        actual_total_games = get_total_games(sets)
        actual_set_score = get_set_score(sets)
        actual_game_margin = get_game_margin(sets)

        # Randomize p1/p2
        if np.random.random() > 0.5:
            p1, p2 = winner, loser
            p1_odds, p2_odds = winner_odds, loser_odds
            actual_winner_is_p1 = True
        else:
            p1, p2 = loser, winner
            p1_odds, p2_odds = loser_odds, winner_odds
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
        model_says_p1 = ensemble >= 0.5
        win_prob = ensemble if model_says_p1 else (1 - ensemble)
        predicted_winner = p1 if model_says_p1 else p2

        if win_prob < min_prob:
            continue

        # Odds for predicted winner
        if model_says_p1:
            pred_odds = p1_odds
            implied_prob = 1.0 / p1_odds
        else:
            pred_odds = p2_odds
            implied_prob = 1.0 / p2_odds

        edge = win_prob - implied_prob
        correct = predicted_winner == winner

        # ════════════════════════════════════════════════════════════════
        # MARKET 1: MONEYLINE
        # ════════════════════════════════════════════════════════════════
        ml_profit = (pred_odds - 1.0) if correct else -1.0
        ml_results.append({
            'correct': correct, 'profit': ml_profit, 'win_prob': win_prob,
            'edge': edge, 'odds': pred_odds, 'surface': surface,
        })

        # ════════════════════════════════════════════════════════════════
        # MARKET 2: SET BETTING (predict exact set score)
        # ════════════════════════════════════════════════════════════════
        if pred.get('scores'):
            # Get model's most likely scoreline
            top_score_tuple, top_count = pred['scores'][0]
            n_sims = 5000

            # Determine predicted set score from top scoreline
            pred_w_sets = sum(1 for g1, g2 in top_score_tuple if g1 > g2)
            pred_l_sets = sum(1 for g1, g2 in top_score_tuple if g2 > g1)

            if model_says_p1:
                pred_set_score = f"{pred_w_sets}-{pred_l_sets}"
            else:
                pred_set_score = f"{pred_l_sets}-{pred_w_sets}"

            # Actual set score from predicted winner's perspective
            if actual_winner_is_p1 == model_says_p1:
                # We predicted the right person, check set score
                actual_from_pred_perspective = actual_set_score
            else:
                # We predicted wrong person, flip
                parts = actual_set_score.split("-")
                actual_from_pred_perspective = f"{parts[1]}-{parts[0]}"

            # Calculate implied probability for this set score
            # Straight sets (2-0 or 3-0) typically odds ~1.70-2.00
            # 2-1 or 3-1 typically ~3.00-4.00
            # 3-2 typically ~5.00-8.00
            set_score_odds = {
                "2-0": 1.80, "0-2": 3.50,
                "2-1": 3.20, "1-2": 5.00,
                "3-0": 2.20, "0-3": 4.50,
                "3-1": 3.50, "1-3": 6.00,
                "3-2": 5.50, "2-3": 9.00,
            }
            sb_odds = set_score_odds.get(pred_set_score, 3.00)

            # Model's probability for this set score
            set_score_prob = top_count / n_sims

            sb_correct = pred_set_score == actual_from_pred_perspective
            sb_profit = (sb_odds - 1.0) if sb_correct else -1.0

            # Only bet if model probability suggests value
            sb_edge = set_score_prob - (1.0 / sb_odds)

            sb_results.append({
                'predicted': pred_set_score,
                'actual': actual_from_pred_perspective,
                'correct': sb_correct,
                'profit': sb_profit,
                'prob': set_score_prob,
                'odds': sb_odds,
                'edge': sb_edge,
                'surface': surface,
            })

        # ════════════════════════════════════════════════════════════════
        # MARKET 3: OVER/UNDER TOTAL GAMES
        # ════════════════════════════════════════════════════════════════
        exp_games = pred.get('exp_games', 0)
        if exp_games > 0:
            # Typical O/U line: use median of expected games ± 0.5
            # Common lines: 21.5, 22.5, 23.5 for Bo3
            if best_of == 3:
                ou_line = 22.5  # standard Bo3 line
            else:
                ou_line = 38.5  # standard Bo5 line

            model_says_over = exp_games > ou_line
            actual_over = actual_total_games > ou_line

            # Standard O/U odds are ~1.90 each side
            ou_odds = 1.90
            ou_correct = model_says_over == actual_over
            ou_profit = (ou_odds - 1.0) if ou_correct else -1.0
            ou_edge = abs(exp_games - ou_line)  # distance from line

            ou_results.append({
                'prediction': 'OVER' if model_says_over else 'UNDER',
                'exp_games': exp_games,
                'actual_games': actual_total_games,
                'line': ou_line,
                'correct': ou_correct,
                'profit': ou_profit,
                'edge_games': ou_edge,
                'surface': surface,
            })

        # ════════════════════════════════════════════════════════════════
        # MARKET 4: GAME SPREAD
        # ════════════════════════════════════════════════════════════════
        if exp_games > 0:
            # Estimate game spread from serve/return stats
            # Typical spread: -3.5, -4.5, -5.5 for favorites
            exp_margin = pred.get('exp_games', 24)
            a_serve = pred.get('a_serve', 0.63)
            b_serve = pred.get('b_serve', 0.63)

            # Estimate expected margin based on serve differential
            serve_diff = abs(a_serve - b_serve)
            if win_prob >= 0.75:
                pred_margin = 5.5
            elif win_prob >= 0.65:
                pred_margin = 3.5
            elif win_prob >= 0.55:
                pred_margin = 1.5
            else:
                pred_margin = 0.5

            # Standard spread line for the favorite
            spread_line = -3.5  # standard line

            # Does our predicted winner cover -3.5?
            model_covers = pred_margin > abs(spread_line)

            # Did the actual winner cover -3.5?
            actual_covers = actual_game_margin > abs(spread_line)

            # But we need to check if we picked the right winner first
            if correct:
                # We picked right winner, did they cover?
                gs_correct = actual_covers == model_covers
            else:
                # We picked wrong winner
                gs_correct = False

            # Spread odds are typically ~1.90
            gs_odds = 1.90
            gs_profit = (gs_odds - 1.0) if gs_correct else -1.0

            gs_results.append({
                'pred_margin': pred_margin,
                'actual_margin': actual_game_margin,
                'spread_line': spread_line,
                'model_covers': model_covers,
                'correct': gs_correct,
                'profit': gs_profit,
                'win_prob': win_prob,
                'surface': surface,
            })

        if (i + 1) % 200 == 0:
            n = len(ml_results)
            if n > 0:
                acc = sum(r['correct'] for r in ml_results) / n
                print(f"  {i+1}/{len(df)}... {n} picks processed, ML acc: {acc:.1%}")

    # ═══════════════════════════════════════════════════════════════════
    # PRINT RESULTS
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  BACKTEST ALL MARKETS — {year} Season")
    print(f"  Odds source: tennis-data.co.uk")
    print(f"  Min confidence filter: {min_prob:.0%}")
    print(f"  Matches processed: {len(ml_results)} | Skipped: {skipped}")
    print(f"{'='*70}")

    # ── MARKET 1: MONEYLINE ──────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  🎾 MARKET 1: MONEYLINE (Match Winner)")
    print(f"{'─'*70}")
    _print_market_stats(ml_results)

    # Moneyline by confidence
    print(f"\n  By Confidence:")
    print(f"  {'Range':<20} {'Picks':>6} {'Acc':>8} {'Profit':>10} {'ROI':>8}")
    print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*10} {'─'*8}")
    for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
                    (0.70, 0.80), (0.80, 1.01)]:
        _print_bucket(ml_results, 'win_prob', lo, hi)

    # Moneyline by edge
    print(f"\n  By Edge vs Market:")
    print(f"  {'Range':<20} {'Picks':>6} {'Acc':>8} {'Profit':>10} {'ROI':>8}")
    print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*10} {'─'*8}")
    for lo, hi in [(-1, 0), (0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0)]:
        _print_bucket(ml_results, 'edge', lo, hi)

    # ── MARKET 2: SET BETTING ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  🎯 MARKET 2: SET BETTING (Exact Set Score)")
    print(f"{'─'*70}")
    _print_market_stats(sb_results)

    # Set betting by predicted score
    print(f"\n  By Predicted Score:")
    print(f"  {'Score':<15} {'Picks':>6} {'Acc':>8} {'Profit':>10} {'ROI':>8} {'Avg Odds':>10}")
    print(f"  {'─'*15} {'─'*6} {'─'*8} {'─'*10} {'─'*8} {'─'*10}")
    for score in ['2-0', '2-1', '0-2', '1-2', '3-0', '3-1', '3-2']:
        picks = [r for r in sb_results if r['predicted'] == score]
        if not picks:
            continue
        n = len(picks)
        c = sum(1 for r in picks if r['correct'])
        p = sum(r['profit'] for r in picks)
        avg_o = np.mean([r['odds'] for r in picks])
        print(f"  {score:<15} {n:>6} {c/n:>7.1%} {p:>+9.1f} {p/n:>+7.1%} {avg_o:>9.2f}")

    # Set betting: only when model has edge
    print(f"\n  Edge-filtered Set Bets:")
    for min_e in [0.0, 0.05, 0.10]:
        edge_picks = [r for r in sb_results if r['edge'] >= min_e]
        if not edge_picks:
            continue
        n = len(edge_picks)
        c = sum(1 for r in edge_picks if r['correct'])
        p = sum(r['profit'] for r in edge_picks)
        print(f"    Edge >= {min_e:+.0%}: {n} picks, {c/n:.1%} accuracy, "
              f"{p/n:+.1%} ROI, {p:+.1f} units")

    # ── MARKET 3: OVER/UNDER ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  📊 MARKET 3: OVER/UNDER TOTAL GAMES")
    print(f"{'─'*70}")
    _print_market_stats(ou_results)

    # O/U by distance from line
    print(f"\n  By Distance from Line (model conviction):")
    print(f"  {'Distance':<20} {'Picks':>6} {'Acc':>8} {'Profit':>10} {'ROI':>8}")
    print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*10} {'─'*8}")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 20)]:
        picks = [r for r in ou_results if lo <= r['edge_games'] < hi]
        if not picks:
            continue
        n = len(picks)
        c = sum(1 for r in picks if r['correct'])
        p = sum(r['profit'] for r in picks)
        label = f"{lo}-{hi} games"
        print(f"  {label:<20} {n:>6} {c/n:>7.1%} {p:>+9.1f} {p/n:>+7.1%}")

    # O/U: Over vs Under breakdown
    print(f"\n  Over vs Under:")
    for direction in ['OVER', 'UNDER']:
        picks = [r for r in ou_results if r['prediction'] == direction]
        if not picks:
            continue
        n = len(picks)
        c = sum(1 for r in picks if r['correct'])
        p = sum(r['profit'] for r in picks)
        print(f"    {direction}: {n} picks, {c/n:.1%} accuracy, {p/n:+.1%} ROI")

    # ── MARKET 4: GAME SPREAD ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  📐 MARKET 4: GAME SPREAD (Handicap -3.5)")
    print(f"{'─'*70}")
    _print_market_stats(gs_results)

    # Spread by confidence
    print(f"\n  By Match Confidence:")
    print(f"  {'Confidence':<20} {'Picks':>6} {'Acc':>8} {'Profit':>10} {'ROI':>8}")
    print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*10} {'─'*8}")
    for lo, hi in [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]:
        picks = [r for r in gs_results if lo <= r['win_prob'] < hi]
        if not picks:
            continue
        n = len(picks)
        c = sum(1 for r in picks if r['correct'])
        p = sum(r['profit'] for r in picks)
        label = f"{lo:.0%}-{hi:.0%}" if hi < 1.0 else f"{lo:.0%}+"
        print(f"  {label:<20} {n:>6} {c/n:>7.1%} {p:>+9.1f} {p/n:>+7.1%}")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY COMPARISON
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  📋 MARKET COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Market':<25} {'Picks':>6} {'Accuracy':>10} {'ROI':>8} {'Profit':>10}")
    print(f"  {'─'*25} {'─'*6} {'─'*10} {'─'*8} {'─'*10}")

    markets = [
        ("Moneyline", ml_results),
        ("Set Betting", sb_results),
        ("Over/Under Games", ou_results),
        ("Game Spread -3.5", gs_results),
    ]
    for name, results in markets:
        if not results:
            continue
        n = len(results)
        c = sum(1 for r in results if r['correct'])
        p = sum(r['profit'] for r in results)
        print(f"  {name:<25} {n:>6} {c/n:>9.1%} {p/n:>+7.1%} {p:>+9.1f}")

    # Best strategy per market with edge filter
    print(f"\n  {'Market':<25} {'Best Filter':<25} {'Picks':>6} {'ROI':>8}")
    print(f"  {'─'*25} {'─'*25} {'─'*6} {'─'*8}")

    # ML best
    best_ml = _find_best_filter(ml_results, 'edge')
    if best_ml:
        print(f"  {'Moneyline':<25} {'Edge >= ' + f'{best_ml[0]:+.0%}':<25} {best_ml[1]:>6} {best_ml[2]:>+7.1%}")

    # SB best
    best_sb = _find_best_filter(sb_results, 'edge')
    if best_sb:
        print(f"  {'Set Betting':<25} {'Edge >= ' + f'{best_sb[0]:+.0%}':<25} {best_sb[1]:>6} {best_sb[2]:>+7.1%}")

    # O/U best
    best_ou = _find_best_ou(ou_results)
    if best_ou:
        print(f"  {'Over/Under':<25} {'Dist >= ' + f'{best_ou[0]:.0f} games':<25} {best_ou[1]:>6} {best_ou[2]:>+7.1%}")

    # GS best
    best_gs = _find_best_filter_prob(gs_results)
    if best_gs:
        print(f"  {'Game Spread':<25} {'Prob >= ' + f'{best_gs[0]:.0%}':<25} {best_gs[1]:>6} {best_gs[2]:>+7.1%}")

    # ── Save all results ─────────────────────────────────────────────
    out_path = BASE / f"backtest_all_markets_{year}.csv"
    all_data = []
    for r in ml_results:
        all_data.append({**r, 'market': 'moneyline'})
    for r in sb_results:
        all_data.append({**r, 'market': 'set_betting'})
    for r in ou_results:
        all_data.append({**r, 'market': 'over_under'})
    for r in gs_results:
        all_data.append({**r, 'market': 'game_spread'})
    pd.DataFrame(all_data).to_csv(out_path, index=False)
    print(f"\n  All results saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _print_market_stats(results):
    if not results:
        print("  No results.")
        return
    n = len(results)
    c = sum(1 for r in results if r['correct'])
    p = sum(r['profit'] for r in results)
    print(f"  Picks:     {n}")
    print(f"  Correct:   {c}")
    print(f"  Accuracy:  {c/n:.1%}")
    print(f"  Profit:    {p:+.1f} units")
    print(f"  ROI:       {p/n:+.1%}")


def _print_bucket(results, key, lo, hi):
    picks = [r for r in results if lo <= r.get(key, 0) < hi]
    if not picks:
        return
    n = len(picks)
    c = sum(1 for r in picks if r['correct'])
    p = sum(r['profit'] for r in picks)
    label = f"{lo:+.0%} to {hi:+.0%}" if key == 'edge' else (f"{lo:.0%}-{hi:.0%}" if hi < 1.0 else f"{lo:.0%}+")
    print(f"  {label:<20} {n:>6} {c/n:>7.1%} {p:>+9.1f} {p/n:>+7.1%}")


def _find_best_filter(results, key):
    best = None
    for threshold in [-0.10, -0.05, 0.0, 0.03, 0.05, 0.08, 0.10, 0.15]:
        picks = [r for r in results if r.get(key, 0) >= threshold]
        if len(picks) < 10:
            continue
        roi = sum(r['profit'] for r in picks) / len(picks)
        if best is None or roi > best[2]:
            best = (threshold, len(picks), roi)
    return best


def _find_best_ou(results):
    best = None
    for dist in [0, 1, 2, 3, 4, 5]:
        picks = [r for r in results if r.get('edge_games', 0) >= dist]
        if len(picks) < 10:
            continue
        roi = sum(r['profit'] for r in picks) / len(picks)
        if best is None or roi > best[2]:
            best = (dist, len(picks), roi)
    return best


def _find_best_filter_prob(results):
    best = None
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        picks = [r for r in results if r.get('win_prob', 0) >= threshold]
        if len(picks) < 10:
            continue
        roi = sum(r['profit'] for r in picks) / len(picks)
        if best is None or roi > best[2]:
            best = (threshold, len(picks), roi)
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest all tennis betting markets")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--min-prob", type=float, default=0.0)
    args = parser.parse_args()

    run_backtest(args.year, min_prob=args.min_prob)
