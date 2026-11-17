"""
backtest_with_odds.py — Backtest with REAL betting odds
========================================================
Uses historical odds from tennis-data.co.uk.

SETUP (one time):
  1. Go to http://www.tennis-data.co.uk/alldata.php
  2. Download the year(s) you want (e.g. 2025.xlsx)
  3. Put them in Tennis_V2/odds_data/

Usage:
    python backtest_with_odds.py                        # 2025 season
    python backtest_with_odds.py --year 2024            # 2024 season
    python backtest_with_odds.py --year 2025 --min-edge 0.05  # only 5%+ edge
    python backtest_with_odds.py --year 2025 --min-prob 0.60  # only 60%+ prob
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
ODDS_DIR = BASE / "odds_data"
sys.path.insert(0, str(BASE))

from build_model import Elo
from predict import predict, FEATURES, ELO


# ═══════════════════════════════════════════════════════════════════════════
# NAME CONVERSION: "Sinner J." → "Jannik Sinner"
# ═══════════════════════════════════════════════════════════════════════════

# Build lookup from your player_features.csv
_PLAYER_LOOKUP = {}

def _build_name_lookup():
    """Build mapping from 'Lastname I.' format to full names in your database."""
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

        # Your DB format: "Jannik Sinner" → first="Jannik", last="Sinner"
        # tennis-data format: "Sinner J."

        initial = first[0] + "."

        # Standard: "Sinner J."
        _PLAYER_LOOKUP[f"{last} {initial}".lower()] = full_name

        # With full first name: "Sinner Jannik"
        _PLAYER_LOOKUP[f"{last} {first}".lower()] = full_name

        # For compound last names, also index by last word
        # "Botic Van De Zandschulp" → also try "Zandschulp B."
        last_parts = last.split()
        if len(last_parts) > 1:
            _PLAYER_LOOKUP[f"{last_parts[-1]} {initial}".lower()] = full_name

        # Two-word first names: "Jan-Lennard Struff"
        # first="Jan-Lennard" → initials "J.L."
        if "-" in first:
            compound_init = ".".join([p[0] for p in first.split("-")]) + "."
            _PLAYER_LOOKUP[f"{last} {compound_init}".lower()] = full_name

    # Also build reverse: full_name.lower() → full_name
    for full_name in all_players:
        _PLAYER_LOOKUP[full_name.strip().lower()] = full_name.strip()

    # Manual overrides for names that don't follow patterns
    manual = {
        # Compound last names (tennis-data uses full compound)
        "auger-aliassime f.": "Felix Auger-Aliassime",
        "van de zandschulp b.": "Botic Van De Zandschulp",
        "van de zandschulp b": "Botic Van De Zandschulp",
        "de minaur a.": "Alex De Minaur",
        "de minaur a": "Alex De Minaur",
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

        # Initials with multiple letters
        "struff j.l.": "Jan-Lennard Struff",
        "struff j.-l.": "Jan-Lennard Struff",
        "cerundolo j.m.": "Juan Manuel Cerundolo",
        "cerundolo f.": "Francisco Cerundolo",

        # Common abbreviation differences
        "de jong j.": "Jesper de Jong",
        "barrios m.": "Tomas Barrios Vera",
    }
    for k, v in manual.items():
        _PLAYER_LOOKUP[k] = v

    print(f"  Name lookup built: {len(_PLAYER_LOOKUP)} entries")


def convert_name(short_name: str) -> str:
    """Convert 'Sinner J.' to 'Jannik Sinner'."""
    _build_name_lookup()
    key = short_name.strip().lower()

    # Direct match
    if key in _PLAYER_LOOKUP:
        return _PLAYER_LOOKUP[key]

    # Strip periods and retry
    key_clean = key.replace(".", "").strip()
    for k, v in _PLAYER_LOOKUP.items():
        if k.replace(".", "").strip() == key_clean:
            return v

    # Parse "Lastname I." format and search by lastname + initial
    parts = short_name.strip().split()
    if len(parts) >= 2:
        # Last element is initial(s), everything before is lastname
        initial_part = parts[-1].replace(".", "").upper()
        lastname_parts = parts[:-1]
        lastname = " ".join(lastname_parts).lower()

        # Search all players whose last name matches and first name starts with initial
        all_players = FEATURES['player'].unique()
        candidates = []
        for full_name in all_players:
            fp = full_name.strip().split()
            if len(fp) < 2:
                continue
            fn_first = fp[0]
            fn_last = " ".join(fp[1:]).lower()

            # Check if lastname matches
            if fn_last == lastname:
                # Check if initial matches
                if fn_first[0].upper() == initial_part[0]:
                    candidates.append(full_name)

        if len(candidates) == 1:
            # Cache it for future
            _PLAYER_LOOKUP[key] = candidates[0]
            return candidates[0]
        elif len(candidates) > 1:
            # Multiple matches — try matching more initials
            if len(initial_part) > 1:
                for c in candidates:
                    c_first = c.split()[0]
                    if "-" in c_first:
                        c_inits = "".join([p[0] for p in c_first.split("-")]).upper()
                        if c_inits == initial_part:
                            _PLAYER_LOOKUP[key] = c
                            return c
            # Return first match as fallback
            _PLAYER_LOOKUP[key] = candidates[0]
            return candidates[0]

    return None  # not found


def load_odds(year: int) -> pd.DataFrame:
    """Load odds data from local file (downloaded manually from tennis-data.co.uk)."""
    ODDS_DIR.mkdir(exist_ok=True)

    # Try xlsx first, then xls
    for ext in ['.xlsx', '.xls']:
        local_file = ODDS_DIR / f"{year}{ext}"
        if local_file.exists():
            print(f"Loading odds: {local_file}")
            engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'
            df = pd.read_excel(local_file, engine=engine)
            print(f"Loaded {len(df)} matches with odds for {year}")
            return df

    print(f"\n  ERROR: No odds file found for {year}.")
    print(f"  Download from: http://www.tennis-data.co.uk/{year}/{year}.xlsx")
    print(f"  Save to: {ODDS_DIR}/{year}.xlsx")
    sys.exit(1)


def run_backtest(year: int, min_edge: float = -999, min_prob: float = 0.0,
                 bookmaker: str = "PSW"):
    """
    Run backtest with real odds.

    bookmaker options (column prefixes in tennis-data.co.uk):
      - B365W/B365L: Bet365
      - PSW/PSL: Pinnacle (sharpest, recommended)
      - AvgW/AvgL: Market average
    """
    df = load_odds(year)

    # Key columns from tennis-data.co.uk:
    # Winner, Loser, Surface, Best of, WRank, LRank, B365W, B365L, PSW, PSL, AvgW, AvgL
    print(f"\nColumns available: {list(df.columns)}")

    # Determine odds columns
    w_odds_col = bookmaker.replace("W", "W") if "W" in bookmaker else f"{bookmaker}W"
    l_odds_col = bookmaker.replace("W", "L") if "W" in bookmaker else f"{bookmaker}L"

    # Fallback to available columns
    if w_odds_col not in df.columns:
        for prefix in ["PS", "B365", "Avg"]:
            if f"{prefix}W" in df.columns:
                w_odds_col, l_odds_col = f"{prefix}W", f"{prefix}L"
                print(f"Using {prefix} odds ({w_odds_col}/{l_odds_col})")
                break

    # Filter to matches with odds
    df = df.dropna(subset=['Winner', 'Loser', w_odds_col, l_odds_col])
    print(f"Matches with odds: {len(df)}")

    # ── Run predictions ──────────────────────────────────────────────────
    results = []
    skipped = 0
    np.random.seed(42)

    for i, (_, row) in enumerate(df.iterrows()):
        winner_raw = str(row['Winner']).strip()
        loser_raw = str(row['Loser']).strip()

        # Convert "Sinner J." → "Jannik Sinner"
        winner = convert_name(winner_raw)
        loser = convert_name(loser_raw)

        if not winner or not loser:
            skipped += 1
            continue

        surface = str(row.get('Surface', 'Hard')).strip()
        best_of = int(row['Best of']) if 'Best of' in row and not pd.isna(row.get('Best of')) else 3
        tourney = str(row.get('Tournament', row.get('Location', '')))

        winner_odds = float(row[w_odds_col])
        loser_odds = float(row[l_odds_col])

        # Randomize p1/p2 to avoid bias
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

        # Calculate edge vs market
        if model_says_p1:
            predicted_odds = p1_odds
            implied_prob = 1.0 / p1_odds
        else:
            predicted_odds = p2_odds
            implied_prob = 1.0 / p2_odds

        edge = win_prob - implied_prob

        # Apply filters
        if win_prob < min_prob:
            continue
        if edge < min_edge:
            continue

        correct = predicted_winner == winner

        # Betting P&L (flat 1 unit bet)
        if correct:
            profit = predicted_odds - 1.0  # win: payout minus stake
        else:
            profit = -1.0  # lose stake

        results.append({
            'tournament': tourney,
            'surface': surface,
            'winner': winner,
            'loser': loser,
            'predicted_winner': predicted_winner,
            'correct': correct,
            'win_prob': round(win_prob, 4),
            'edge': round(edge, 4),
            'predicted_odds': round(predicted_odds, 2),
            'implied_prob': round(implied_prob, 4),
            'profit': round(profit, 2),
            'elo_prob': round(pred['elo_prob'] if model_says_p1 else 1 - pred['elo_prob'], 4),
            'mc_prob': round(pred['mc_prob'] if model_says_p1 else 1 - pred['mc_prob'], 4),
        })

        if (i + 1) % 100 == 0:
            n = len(results)
            if n > 0:
                acc = sum(r['correct'] for r in results) / n
                roi = sum(r['profit'] for r in results) / n
                print(f"  {i+1}/{len(df)}... {n} picks, {acc:.1%} accuracy, {roi:+.1%} ROI")

    if not results:
        print("\nNo predictions generated.")
        return

    # ── Print Results ────────────────────────────────────────────────────
    rdf = pd.DataFrame(results)
    total = len(rdf)
    correct = rdf['correct'].sum()
    accuracy = correct / total
    total_profit = rdf['profit'].sum()
    roi = total_profit / total

    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS — {year} Season")
    print(f"  Odds source: tennis-data.co.uk ({w_odds_col}/{l_odds_col})")
    print(f"  Filters: min_prob={min_prob:.0%}, min_edge={min_edge:+.0%}")
    print(f"{'='*70}")

    print(f"\n  📊 ACCURACY")
    print(f"  Picks:     {total}")
    print(f"  Correct:   {correct}")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Skipped:   {skipped}")

    print(f"\n  💰 BETTING (flat 1 unit per pick)")
    print(f"  Total wagered:  {total} units")
    print(f"  Total profit:   {total_profit:+.1f} units")
    print(f"  ROI:            {roi:+.1%}")
    print(f"  Avg odds:       {rdf['predicted_odds'].mean():.2f}")
    print(f"  Avg edge:       {rdf['edge'].mean():+.1%}")

    # ── By edge bucket ───────────────────────────────────────────────────
    print(f"\n  {'Edge Range':<20} {'Picks':>6} {'Acc':>8} {'Profit':>10} {'ROI':>8}")
    print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*10} {'─'*8}")

    edge_buckets = [(-1, 0.00), (0.00, 0.03), (0.03, 0.06), (0.06, 0.10),
                    (0.10, 0.15), (0.15, 0.25), (0.25, 1.0)]
    for lo, hi in edge_buckets:
        mask = (rdf['edge'] >= lo) & (rdf['edge'] < hi)
        b = rdf[mask]
        if len(b) == 0:
            continue
        b_acc = b['correct'].mean()
        b_profit = b['profit'].sum()
        b_roi = b_profit / len(b)
        label = f"{lo:+.0%} to {hi:+.0%}"
        print(f"  {label:<20} {len(b):>6} {b_acc:>7.1%} {b_profit:>+9.1f} {b_roi:>+7.1%}")

    # ── By confidence bucket ─────────────────────────────────────────────
    print(f"\n  {'Confidence':<20} {'Picks':>6} {'Acc':>8} {'Profit':>10} {'ROI':>8}")
    print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*10} {'─'*8}")

    prob_buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
                    (0.70, 0.80), (0.80, 1.01)]
    for lo, hi in prob_buckets:
        mask = (rdf['win_prob'] >= lo) & (rdf['win_prob'] < hi)
        b = rdf[mask]
        if len(b) == 0:
            continue
        b_acc = b['correct'].mean()
        b_profit = b['profit'].sum()
        b_roi = b_profit / len(b)
        label = f"{lo:.0%}-{hi:.0%}" if hi < 1.0 else f"{lo:.0%}+"
        print(f"  {label:<20} {len(b):>6} {b_acc:>7.1%} {b_profit:>+9.1f} {b_roi:>+7.1%}")

    # ── By surface ───────────────────────────────────────────────────────
    print(f"\n  {'Surface':<20} {'Picks':>6} {'Acc':>8} {'ROI':>8}")
    print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*8}")
    for surf in sorted(rdf['surface'].unique()):
        s = rdf[rdf['surface'] == surf]
        s_acc = s['correct'].mean()
        s_roi = s['profit'].sum() / len(s)
        print(f"  {surf:<20} {len(s):>6} {s_acc:>7.1%} {s_roi:>+7.1%}")

    # ── Calibration ──────────────────────────────────────────────────────
    print(f"\n  Calibration:")
    print(f"  {'Predicted':<20} {'Actual Win%':>12} {'N':>6}")
    print(f"  {'─'*20} {'─'*12} {'─'*6}")
    for lo, hi in [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]:
        mask = (rdf['win_prob'] >= lo) & (rdf['win_prob'] < hi)
        b = rdf[mask]
        if len(b) == 0:
            continue
        actual = b['correct'].mean()
        label = f"{lo:.0%}-{hi:.0%}" if hi < 1.0 else f"{lo:.0%}+"
        print(f"  {label:<20} {actual:>11.1%} {len(b):>6}")

    # ── The money line: edge-filtered results ────────────────────────────
    print(f"\n{'='*70}")
    print(f"  💎 EDGE PICKS SUMMARY (what you'd tell clients)")
    print(f"{'='*70}")

    for min_e in [0.0, 0.03, 0.05, 0.08, 0.10]:
        edge_picks = rdf[rdf['edge'] >= min_e]
        if len(edge_picks) == 0:
            continue
        e_acc = edge_picks['correct'].mean()
        e_profit = edge_picks['profit'].sum()
        e_roi = e_profit / len(edge_picks)
        print(f"  Edge >= {min_e:+.0%}: {len(edge_picks)} picks, "
              f"{e_acc:.1%} accuracy, {e_roi:+.1%} ROI, "
              f"{e_profit:+.1f} units profit")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = BASE / f"backtest_odds_{year}.csv"
    rdf.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest with real betting odds")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--min-edge", type=float, default=-999,
                        help="Only bet when edge >= this value (e.g. 0.05)")
    parser.add_argument("--min-prob", type=float, default=0.0,
                        help="Only bet when model prob >= this value")
    parser.add_argument("--bookmaker", type=str, default="PSW",
                        help="Odds source: PSW (Pinnacle), B365W (Bet365), AvgW (market avg)")
    args = parser.parse_args()

    run_backtest(args.year, min_edge=args.min_edge, min_prob=args.min_prob,
                 bookmaker=args.bookmaker)
