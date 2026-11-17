"""
predict.py — Tennis Match Predictor
=====================================
Predicts any ATP match using real data from TennisMyLife.
Combines Elo + serve/return stats + point-based Markov + Monte Carlo.

Usage:
    python predict.py "Jannik Sinner" "Carlos Alcaraz"
    python predict.py "Novak Djokovic" "Rafael Nadal" --clay --best_of 5
    python predict.py --top 30
    python predict.py --search "Fritz"
    python predict.py --today                # Predict today's matches (requires ongoing_tourneys.csv)
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
sys.path.insert(0, str(BASE))
from tennis_math import p_match, sim_match
from build_model import Elo

# ─── Load data ─────────────────────────────────────────────────
print("Loading player database...")
FEATURES = pd.read_csv(MODEL_DIR / 'player_features.csv')
FEATURES['date'] = pd.to_datetime(FEATURES['date'])

with open(MODEL_DIR / 'elo.pkl', 'rb') as f:
    ELO = pickle.load(f)


def player_stats(name: str) -> dict:
    """Latest rolling stats for a player."""
    pdf = FEATURES[FEATURES['player'] == name].sort_values('date')
    if len(pdf) == 0:
        close = [p for p in FEATURES['player'].unique() if name.lower() in p.lower()]
        if close:
            print(f"  '{name}' not found. Similar: {close[:5]}")
        else:
            print(f"  '{name}' not found.")
        return None

    last = pdf.iloc[-1]
    return {
        'name': name,
        'matches': len(pdf),
        'last_date': str(last['date'].date()),
        'rank': last['rank'] if not pd.isna(last.get('rank')) else 999,
        'elo': ELO.r(name),
        'elo_hard': ELO.r(name, 'Hard'),
        'elo_clay': ELO.r(name, 'Clay'),
        'elo_grass': ELO.r(name, 'Grass'),
        'serve': last.get('r_serve_pct', np.nan),
        'return': last.get('r_return_pct', np.nan),
        '1st_in': last.get('r_1st_in', np.nan),
        '1st_won': last.get('r_1st_won', np.nan),
        '2nd_won': last.get('r_2nd_won', np.nan),
        'bp_saved': last.get('r_bp_saved', np.nan),
        'bp_conv': last.get('r_bp_conv', np.nan),
        'surf_serve_hard': last.get('s_Hard_serve_pct', np.nan),
        'surf_return_hard': last.get('s_Hard_return_pct', np.nan),
        'surf_serve_clay': last.get('s_Clay_serve_pct', np.nan),
        'surf_serve_grass': last.get('s_Grass_serve_pct', np.nan),
    }


# ─── Predict ───────────────────────────────────────────────────
def predict(a_name: str, b_name: str, surface: str = 'Hard',
            best_of: int = 3, n_sims: int = 30000) -> dict:
    a = player_stats(a_name)
    b = player_stats(b_name)
    if not a or not b:
        return None

    # Serve % (surface-specific if available)
    surf_key = f'surf_serve_{surface.lower()}'
    a_serve = a.get(surf_key, np.nan)
    if pd.isna(a_serve): a_serve = a['serve']
    if pd.isna(a_serve): a_serve = 0.63

    b_serve = b.get(surf_key, np.nan)
    if pd.isna(b_serve): b_serve = b['serve']
    if pd.isna(b_serve): b_serve = 0.63

    # Effective return (blend own return + opponent serve)
    a_ret = a['return'] if not pd.isna(a['return']) else 0.35
    b_ret = b['return'] if not pd.isna(b['return']) else 0.35
    a_eff_ret = 0.6 * a_ret + 0.4 * (1 - b_serve)
    b_eff_ret = 0.6 * b_ret + 0.4 * (1 - a_serve)

    # Elo
    elo_key = f'elo_{surface.lower()}'
    a_elo = a.get(elo_key, a['elo'])
    b_elo = b.get(elo_key, b['elo'])
    elo_diff = a_elo - b_elo
    elo_prob = 1 / (1 + 10 ** (-elo_diff / 400))

    # Analytical
    analytical = p_match(a_serve, a_eff_ret, best_of=best_of)

    # Monte Carlo
    mc = sim_match(a_serve, b_serve, best_of=best_of, n=n_sims)

    # Ensemble
    ensemble = 0.40 * elo_prob + 0.30 * mc['p_win'] + 0.30 * analytical
    ensemble = float(np.clip(ensemble, 0.02, 0.98))

    return {
        'a': a, 'b': b,
        'surface': surface, 'best_of': best_of,
        'a_serve': a_serve, 'b_serve': b_serve,
        'a_eff_ret': a_eff_ret, 'b_eff_ret': b_eff_ret,
        'elo_prob': elo_prob,
        'analytical': analytical,
        'mc_prob': mc['p_win'],
        'ensemble': ensemble,
        'exp_games': mc['avg_games'],
        'std_games': mc['std_games'],
        'exp_sets': mc['avg_sets'],
        'scores': mc['scores'],
    }


def display(p):
    if not p: return
    a, b = p['a'], p['b']
    a_s, b_s = a['name'].split()[-1], b['name'].split()[-1]

    print(f"\n{'='*65}")
    print(f"  {a['name']} vs {b['name']}")
    print(f"  {p['surface']} | Best of {p['best_of']}")
    print(f"{'='*65}")

    print(f"\n  {'Stat':<30} {a['name']:>15} {b['name']:>15}")
    print(f"  {'─'*30} {'─'*15} {'─'*15}")
    print(f"  {'Elo (' + p['surface'] + ')':<30} {a.get('elo_'+p['surface'].lower(), 0):>15.0f} {b.get('elo_'+p['surface'].lower(), 0):>15.0f}")
    print(f"  {'Rank':<30} {a['rank']:>15.0f} {b['rank']:>15.0f}")
    print(f"  {'Serve pts won %':<30} {p['a_serve']:>14.1%} {p['b_serve']:>14.1%}")
    print(f"  {'Eff. return %':<30} {p['a_eff_ret']:>14.1%} {p['b_eff_ret']:>14.1%}")
    print(f"  {'Matches in DB':<30} {a['matches']:>15} {b['matches']:>15}")
    print(f"  {'Last match':<30} {a['last_date']:>15} {b['last_date']:>15}")

    print(f"\n  {'Model':<20} {a_s:>20} {b_s:>20}")
    print(f"  {'─'*20} {'─'*20} {'─'*20}")
    print(f"  {'Elo':<20} {p['elo_prob']:>19.1%} {1-p['elo_prob']:>19.1%}")
    print(f"  {'Analytical':<20} {p['analytical']:>19.1%} {1-p['analytical']:>19.1%}")
    print(f"  {'Simulation':<20} {p['mc_prob']:>19.1%} {1-p['mc_prob']:>19.1%}")
    print(f"  {'ENSEMBLE':<20} {p['ensemble']:>19.1%} {1-p['ensemble']:>19.1%}")

    print(f"\n  Expected games: {p['exp_games']:.1f} ± {p['std_games']:.1f}")
    print(f"  Expected sets:  {p['exp_sets']:.2f}")

    print(f"\n  Most likely scorelines:")
    for score, count in p['scores']:
        pct = count / 30000 * 100
        s = " ".join([f"{g1}-{g2}" for g1, g2 in score])
        print(f"    {s}: {pct:.1f}%")
    # Total games distribution
    from collections import defaultdict
    games_dist = defaultdict(int)
    total_sims = 30000
    for score, count in p['scores']:
        total_games = sum(g1 + g2 for g1, g2 in score)
        games_dist[total_games] += count

    print(f"\n  Total Games Distribution:")
    print(f"  {'Games':<10} {'Prob':>8} {'Cumul':>8}  {'Bar'}")
    print(f"  {'─'*10} {'─'*8} {'─'*8}  {'─'*20}")
    cumul = 0.0
    for games in sorted(games_dist.keys()):
        pct = games_dist[games] / total_sims * 100
        cumul += pct
        bar = "█" * int(pct * 2)
        print(f"  {games:<10} {pct:>7.1f}% {cumul:>7.1f}%  {bar}")


def top_players(n=30, surface=None):
    ratings = sorted(ELO.ratings.items(), key=lambda x: -x[1])
    print(f"\n  Top {n} by Elo")
    print(f"  {'#':>3} {'Player':<28} {'Elo':>6} {'Hard':>6} {'Clay':>6} {'Grass':>6} {'Sv%':>6} {'Rt%':>6}")
    print(f"  {'─'*3} {'─'*28} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for i, (p, r) in enumerate(ratings[:n], 1):
        s = player_stats(p)
        if not s: continue
        sv = f"{s['serve']:.1%}" if not pd.isna(s['serve']) else " N/A"
        rt = f"{s['return']:.1%}" if not pd.isna(s['return']) else " N/A"
        print(f"  {i:>3} {p:<28} {r:>6.0f} {s['elo_hard']:>6.0f} {s['elo_clay']:>6.0f} {s['elo_grass']:>6.0f} {sv:>6} {rt:>6}")


def search(query):
    matches = [p for p in FEATURES['player'].unique() if query.lower() in p.lower()]
    if not matches:
        print(f"  No players found for '{query}'")
    else:
        print(f"\n  {len(matches)} matches for '{query}':")
        for p in sorted(matches)[:20]:
            print(f"    {p:<30} Elo: {ELO.r(p):.0f}")


# ─── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or '--help' in args:
        print("""
  Usage:
    python predict.py "Player A" "Player B"
    python predict.py "Player A" "Player B" --clay
    python predict.py "Player A" "Player B" --best_of 5
    python predict.py --top 30
    python predict.py --search "Djokovic"
        """)
    elif '--top' in args:
        n = int(args[args.index('--top') + 1]) if len(args) > args.index('--top') + 1 else 30
        top_players(n)
    elif '--search' in args:
        search(args[args.index('--search') + 1])
    elif len(args) >= 2 and not args[0].startswith('--'):
        surface = 'Clay' if '--clay' in args else 'Grass' if '--grass' in args else 'Hard'
        bo = int(args[args.index('--best_of') + 1]) if '--best_of' in args else 3
        display(predict(args[0], args[1], surface=surface, best_of=bo))
