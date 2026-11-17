"""
sim_montecarlo.py — Simulate Rolex Monte-Carlo Masters 2026
=============================================================
Full draw from official PDF. This tournament is COMPLETED.
Sinner won, beating Alcaraz in the final 6-1 6-4.

Can be used to:
  - Backtest: compare model predictions vs actual results
  - Validate: did the model correctly identify Sinner as favorite?

Usage:
    python sim_montecarlo.py                  # deterministic
    python sim_montecarlo.py --mc 500         # Monte Carlo
    python sim_montecarlo.py --backtest       # compare predictions vs actual results
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from build_model import Elo
from predict import predict, player_stats, ELO, FEATURES

SURFACE = "Clay"

# ═══════════════════════════════════════════════════════════════════════════
# R64 MATCHUPS (Round of 64 — first round for this tournament)
# Monte-Carlo is a 56-draw Masters (no R128). Top 8 seeds get byes.
# ═══════════════════════════════════════════════════════════════════════════

R64_MATCHUPS = [
    # ── ALCARAZ QUARTER (Top Left) ────────────────────────────────────────
    # Alcaraz section
    ("Sebastian Baez", "Stan Wawrinka"),               # 0  → Baez won 7-5 7-5
    ("Tomas Martin Etcheverry", "Grigor Dimitrov"),     # 1  → Etcheverry won 6-4 2-6 6-3
    ("Terence Atmane", "Ethan Quinn"),                  # 2  LL → Atmane won 6-1 6-4
    ("Jiri Lehecka", "Emilio Nava"),                    # 3  11 vs Q → Lehecka won 7-6 6-7 6-2
    ("Marton Fucsovics", "Alejandro Tabilo"),           # 4  → Tabilo won 6-4 6-3
    ("Gael Monfils", "Tallon Griekspoor"),              # 5  WC vs → Monfils won 6-7 6-1 6-4

    # Musetti section
    ("Valentin Vacherot", "Juan Manuel Cerundolo"),     # 6  → Vacherot won 7-6 7-5
    ("Damir Dzumhur", "Fabian Marozsan"),               # 7  → Marozsan won 6-2 6-1
    ("Hubert Hurkacz", "Luciano Darderi"),              # 8  PR vs 15 → Hurkacz won 7-6 5-7 6-1
    ("Flavio Cobolli", "Francisco Comesana"),           # 9  10 vs Q → Cobolli won 7-5 2-6 6-3
    ("Denis Shapovalov", "Alexander Blockx"),           # 10 → Blockx won 6-4 4-6 6-3
    ("Cameron Norrie", "Miomir Kecmanovic"),            # 11 → Norrie won (walkover implied by score 6-2 4-6 7-6)

    # ── SINNER QUARTER (Top Right) ────────────────────────────────────────
    # Medvedev section
    ("Roberto Bautista Agut", "Matteo Berrettini"),     # 12 LL vs WC → Berrettini won 4-0 RET
    ("Joao Fonseca", "Gabriel Diallo"),                 # 13 → Fonseca won 6-2 6-3
    ("Arthur Rinderknech", "Karen Khachanov"),          # 14 vs 12 → Rinderknech won 7-5 7-2 (walkover?)
    # Rublev section
    ("Andrey Rublev", "Nuno Borges"),                   # 15 13 vs → Rublev won 6-4 1-6 6-1
    ("Zizou Bergs", "Adrian Mannarino"),                # 16 → Bergs won 6-4 6-3
    ("Cristian Garin", "Matteo Arnaldi"),               # 17 Q vs LL → Garin won 6-2 6-4

    # FAA section
    ("Marin Cilic", "Alexander Shevchenko"),            # 18 vs Q → Cilic won 6-1 6-3
    ("Corentin Moutet", "Alexandre Muller"),            # 19 vs Q → Moutet won 6-4 6-1
    ("Alexei Popyrin", "Casper Ruud"),                  # 20 vs 9 → Ruud won 6-3 6-4
    # Sinner section
    ("Francisco Cerundolo", "Stefanos Tsitsipas"),      # 21 16 vs → Cerundolo won 7-5 6-4
    ("Daniel Altmaier", "Tomas Machac"),                # 22 → Machac won 6-4 1-6 6-3
    ("Moise Kouame", "Ugo Humbert"),                    # 23 WC vs → Humbert won 6-3 7-5
]

# ═══════════════════════════════════════════════════════════════════════════
# R32 BRACKET: (seed_with_bye_or_R64_winner_index, same)
# Seeds with byes enter at R32
# ═══════════════════════════════════════════════════════════════════════════

R32_BRACKET = [
    # ── ALCARAZ HALF ──────────────────────────────────────────────────────
    ("Carlos Alcaraz", 0),              # 1 vs Baez/Wawrinka winner
    (1, 2),                              # Etcheverry/Dimitrov vs Atmane/Quinn
    (3, 4),                              # Lehecka/Nava vs Tabilo/Fucsovics
    (5, "Alexander Bublik"),             # Monfils/Griekspoor vs 8

    ("Lorenzo Musetti", 6),              # 4 vs Vacherot/JM Cerundolo
    (7, 8),                              # Marozsan/Dzumhur vs Hurkacz/Darderi
    (9, 10),                             # Cobolli/Comesana vs Blockx/Shapovalov
    (11, "Alex De Minaur"),              # Norrie/Kecmanovic vs 5

    # ── SINNER HALF ───────────────────────────────────────────────────────
    ("Daniil Medvedev", 12),             # 7 vs Berrettini/Bautista
    (13, 14),                            # Fonseca/Diallo vs Rinderknech/Khachanov
    (15, 16),                            # Rublev/Borges vs Bergs/Mannarino
    (17, "Alexander Zverev"),            # Garin/Arnaldi vs 3

    ("Felix Auger-Aliassime", 18),       # 6 vs Cilic/Shevchenko
    (19, 20),                            # Moutet/Muller vs Ruud/Popyrin
    (21, 22),                            # Cerundolo/Tsitsipas vs Machac/Altmaier
    (23, "Jannik Sinner"),               # Humbert/Kouame vs 2
]

# ═══════════════════════════════════════════════════════════════════════════
# ACTUAL RESULTS (for backtest comparison)
# ═══════════════════════════════════════════════════════════════════════════

ACTUAL_RESULTS = {
    'R32': [
        'Carlos Alcaraz', 'Tomas Martin Etcheverry', 'Jiri Lehecka', 'Alexander Bublik',
        'Valentin Vacherot', 'Hubert Hurkacz', 'Alexander Blockx', 'Alex De Minaur',
        'Matteo Berrettini', 'Joao Fonseca', 'Zizou Bergs', 'Alexander Zverev',
        'Felix Auger-Aliassime', 'Casper Ruud', 'Tomas Machac', 'Jannik Sinner',
    ],
    'R16': [
        'Carlos Alcaraz', 'Jiri Lehecka', 'Valentin Vacherot', 'Alex De Minaur',
        'Joao Fonseca', 'Alexander Zverev', 'Felix Auger-Aliassime', 'Jannik Sinner',
    ],
    'QF': [
        'Carlos Alcaraz', 'Valentin Vacherot', 'Alexander Zverev', 'Jannik Sinner',
    ],
    'SF': [
        'Carlos Alcaraz', 'Jannik Sinner',
    ],
    'Final': [
        'Jannik Sinner',  # Won 6-1 6-4
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def get_prob(p1: str, p2: str) -> float:
    try:
        result = predict(p1, p2, surface=SURFACE, best_of=3, n_sims=5000)
        if result:
            return result['ensemble']
    except:
        pass
    return 0.5


def run_deterministic():
    def sim_fn(p1, p2):
        prob = get_prob(p1, p2)
        return p1 if prob >= 0.5 else p2

    # ── R64 ──
    print(f"\n{'='*70}")
    print(f"  ROUND OF 64")
    print(f"{'='*70}")

    r64_winners = []
    for i, (p1, p2) in enumerate(R64_MATCHUPS):
        winner = sim_fn(p1, p2)
        prob = get_prob(p1, p2)
        w_prob = prob if winner == p1 else 1 - prob
        print(f"  [{i:>2}] {p1:<30} vs {p2:<30} → {winner} ({w_prob:.0%})")
        r64_winners.append(winner)

    # ── R32 ──
    print(f"\n{'='*70}")
    print(f"  ROUND OF 32 (seeds enter)")
    print(f"{'='*70}")

    r32_matches = []
    for top, bottom in R32_BRACKET:
        p1 = r64_winners[top] if isinstance(top, int) else top
        p2 = r64_winners[bottom] if isinstance(bottom, int) else bottom
        r32_matches.append((p1, p2))

    current = []
    for p1, p2 in r32_matches:
        winner = sim_fn(p1, p2)
        prob = get_prob(p1, p2)
        w_prob = prob if winner == p1 else 1 - prob
        print(f"  {p1:<30} vs {p2:<30} → {winner} ({w_prob:.0%})")
        current.append(winner)

    # ── R16 → Final ──
    round_names = ["ROUND OF 16", "QUARTERFINALS", "SEMIFINALS", "FINAL"]
    for rname in round_names:
        if len(current) <= 1:
            break
        print(f"\n{'='*70}")
        print(f"  {rname}")
        print(f"{'='*70}")
        next_round = []
        for i in range(0, len(current), 2):
            p1, p2 = current[i], current[i + 1]
            winner = sim_fn(p1, p2)
            prob = get_prob(p1, p2)
            w_prob = prob if winner == p1 else 1 - prob
            print(f"  {p1:<30} vs {p2:<30} → {winner} ({w_prob:.0%})")
            next_round.append(winner)
        current = next_round

    print(f"\n{'='*70}")
    print(f"  🏆 CHAMPION: {current[0]}")
    print(f"{'='*70}")


def run_monte_carlo(n_sims: int = 500):
    print(f"\n  Running {n_sims} Monte Carlo simulations on Clay...")
    print(f"  Pre-computing probabilities...\n")

    prob_cache = {}

    def cached_prob(p1, p2):
        key = (p1, p2)
        if key not in prob_cache:
            prob_cache[key] = get_prob(p1, p2)
            prob_cache[(p2, p1)] = 1 - prob_cache[key]
        return prob_cache[key]

    def sim_fn(p1, p2):
        prob = cached_prob(p1, p2)
        return p1 if np.random.random() < prob else p2

    champions = defaultdict(int)
    finalists = defaultdict(int)
    semifinalists = defaultdict(int)
    quarterfinalists = defaultdict(int)

    for sim in range(n_sims):
        # R64
        r64_winners = []
        for p1, p2 in R64_MATCHUPS:
            r64_winners.append(sim_fn(p1, p2))

        # R32
        r32_matches = []
        for top, bottom in R32_BRACKET:
            p1 = r64_winners[top] if isinstance(top, int) else top
            p2 = r64_winners[bottom] if isinstance(bottom, int) else bottom
            r32_matches.append((p1, p2))

        current = [sim_fn(p1, p2) for p1, p2 in r32_matches]

        # R16 → Final
        while len(current) > 1:
            next_round = []
            for i in range(0, len(current), 2):
                if i + 1 >= len(current):
                    next_round.append(current[i])
                else:
                    next_round.append(sim_fn(current[i], current[i + 1]))

            if len(next_round) == 4:
                for p in next_round:
                    quarterfinalists[p] += 1
            elif len(next_round) == 2:
                for p in next_round:
                    semifinalists[p] += 1
            elif len(next_round) == 1:
                # Both finalists
                for p in current[-2:]:
                    finalists[p] += 1
                champions[next_round[0]] += 1

            current = next_round

        if (sim + 1) % 50 == 0:
            top = sorted(champions.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join([f"{p}: {c/(sim+1):.1%}" for p, c in top])
            print(f"    Sim {sim+1}/{n_sims} — Top: {top_str}")

    # ── Results ──
    print(f"\n{'='*70}")
    print(f"  ROLEX MONTE-CARLO MASTERS 2026 — MONTE CARLO ({n_sims} sims, Clay)")
    print(f"{'='*70}")

    print(f"\n  🏆 CHAMPION PROBABILITY")
    print(f"  {'Player':<30} {'Wins':>6} {'Prob':>8}  {'Bar'}")
    print(f"  {'─'*30} {'─'*6} {'─'*8}  {'─'*25}")
    sorted_champs = sorted(champions.items(), key=lambda x: -x[1])
    max_c = sorted_champs[0][1] if sorted_champs else 1
    for player, count in sorted_champs[:20]:
        prob = count / n_sims
        bar = "█" * int(25 * count / max_c)
        print(f"  {player:<30} {count:>6} {prob:>7.1%}  {bar}")

    print(f"\n  🎾 FINALIST PROBABILITY")
    print(f"  {'Player':<30} {'Times':>6} {'Prob':>8}")
    print(f"  {'─'*30} {'─'*6} {'─'*8}")
    for player, count in sorted(finalists.items(), key=lambda x: -x[1])[:15]:
        print(f"  {player:<30} {count:>6} {count/n_sims:>7.1%}")

    print(f"\n  🏅 SEMIFINALIST PROBABILITY")
    print(f"  {'Player':<30} {'Times':>6} {'Prob':>8}")
    print(f"  {'─'*30} {'─'*6} {'─'*8}")
    for player, count in sorted(semifinalists.items(), key=lambda x: -x[1])[:15]:
        print(f"  {player:<30} {count:>6} {count/n_sims:>7.1%}")

    print(f"\n  📊 QUARTERFINALIST PROBABILITY")
    print(f"  {'Player':<30} {'Times':>6} {'Prob':>8}")
    print(f"  {'─'*30} {'─'*6} {'─'*8}")
    for player, count in sorted(quarterfinalists.items(), key=lambda x: -x[1])[:20]:
        print(f"  {player:<30} {count:>6} {count/n_sims:>7.1%}")

    print(f"\n  💰 OUTRIGHT WINNER — FAIR ODDS")
    print(f"  {'Player':<30} {'Model':>8} {'Fair Odds':>10}")
    print(f"  {'─'*30} {'─'*8} {'─'*10}")
    for player, count in sorted_champs[:10]:
        prob = count / n_sims
        if prob > 0:
            print(f"  {player:<30} {prob:>7.1%} {1/prob:>9.1f}")


def run_backtest():
    """Compare model predictions vs actual Monte-Carlo 2026 results."""
    print(f"\n{'='*70}")
    print(f"  BACKTEST: MODEL vs ACTUAL RESULTS")
    print(f"  Rolex Monte-Carlo Masters 2026 (Clay)")
    print(f"{'='*70}")

    # Simulate R64
    r64_winners_model = []
    r64_correct = 0
    r64_total = 0

    print(f"\n  ROUND OF 64")
    print(f"  {'Match':<50} {'Model':>12} {'Actual':>12} {'✓':>3}")
    print(f"  {'─'*50} {'─'*12} {'─'*12} {'─'*3}")

    for i, (p1, p2) in enumerate(R64_MATCHUPS):
        prob = get_prob(p1, p2)
        model_pick = p1 if prob >= 0.5 else p2
        r64_winners_model.append(model_pick)

        # Find actual winner from R32 bracket
        # The R32 results tell us who advanced from R64
        actual_r32 = ACTUAL_RESULTS.get('R32', [])
        actual = None
        if p1.split()[-1] in [x.split()[-1] for x in actual_r32]:
            actual = p1
        elif p2.split()[-1] in [x.split()[-1] for x in actual_r32]:
            actual = p2

        correct = "✓" if model_pick == actual else "✗" if actual else "?"
        if actual:
            r64_total += 1
            if model_pick == actual:
                r64_correct += 1

        m_short = model_pick.split()[-1]
        a_short = actual.split()[-1] if actual else "?"
        matchup = f"{p1.split()[-1]} vs {p2.split()[-1]}"
        print(f"  {matchup:<50} {m_short:>12} {a_short:>12} {correct:>3}")

    # R32 onwards
    for rname, actual_winners in ACTUAL_RESULTS.items():
        if rname == 'R32':
            continue
        # Get the round before
        prev_round_key = {'R16': 'R32', 'QF': 'R16', 'SF': 'QF', 'Final': 'SF'}
        prev = ACTUAL_RESULTS.get(prev_round_key.get(rname, ''), [])
        if len(prev) < 2:
            continue

        print(f"\n  {rname.upper()}")
        print(f"  {'Match':<50} {'Model':>12} {'Actual':>12} {'✓':>3}")
        print(f"  {'─'*50} {'─'*12} {'─'*12} {'─'*3}")

        round_correct = 0
        round_total = 0
        for i in range(0, len(prev), 2):
            if i + 1 >= len(prev):
                break
            p1, p2 = prev[i], prev[i + 1]
            prob = get_prob(p1, p2)
            model_pick = p1 if prob >= 0.5 else p2
            m_prob = prob if model_pick == p1 else 1 - prob

            actual = None
            for aw in actual_winners:
                if aw.split()[-1] in [p1.split()[-1], p2.split()[-1]]:
                    actual = p1 if aw.split()[-1] == p1.split()[-1] else p2
                    break

            correct = "✓" if model_pick == actual else "✗" if actual else "?"
            if actual:
                round_total += 1
                if model_pick == actual:
                    round_correct += 1
                    r64_correct += 1
                r64_total += 1

            m_short = model_pick.split()[-1]
            a_short = actual.split()[-1] if actual else "?"
            matchup = f"{p1.split()[-1]} vs {p2.split()[-1]}"
            print(f"  {matchup:<50} {m_short:>12} ({m_prob:.0%}) {a_short:>8} {correct:>3}")

    print(f"\n{'='*70}")
    print(f"  OVERALL ACCURACY: {r64_correct}/{r64_total} = {r64_correct/r64_total:.1%}" if r64_total > 0 else "  No matches to compare")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte-Carlo Masters 2026 Simulator")
    parser.add_argument("--mc", type=int, default=0,
                        help="Monte Carlo simulations (0 = deterministic)")
    parser.add_argument("--backtest", action="store_true",
                        help="Compare model predictions vs actual results")
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    elif args.mc > 0:
        np.random.seed(42)
        run_monte_carlo(n_sims=args.mc)
    else:
        run_deterministic()
