"""
tennis_math.py — Point-Based Probability Model
=================================================
Given p (serve point win %), computes:
  P(game) → P(tiebreak) → P(set) → P(match)

Two methods:
  - Analytical: exact closed-form + DP
  - Monte Carlo: simulated point-by-point (gives scorelines)
"""

import numpy as np
from functools import lru_cache


def p_game(p: float) -> float:
    """P(server wins game) given p = P(server wins a point)."""
    q = 1 - p
    p_deuce = p**2 / (p**2 + q**2)
    return p**4 * (1 + 4*q + 10*q**2) + 20 * p**3 * q**3 * p_deuce


def p_tiebreak(p_serve: float, p_return: float) -> float:
    """P(player A wins tiebreak). A serves first."""
    @lru_cache(maxsize=None)
    def dp(a, b, pt):
        if a >= 7 and a - b >= 2: return 1.0
        if b >= 7 and b - a >= 2: return 0.0
        if a > 20 or b > 20: return 0.5

        if pt == 1:
            serving = True
        else:
            serving = ((pt - 2) // 2) % 2 == 1

        p = p_serve if serving else p_return
        return p * dp(a+1, b, pt+1) + (1-p) * dp(a, b+1, pt+1)

    r = dp(0, 0, 1)
    dp.cache_clear()
    return r


def p_set(p_serve: float, p_return: float) -> float:
    """P(A wins set). A serves first game."""
    hold = p_game(p_serve)
    brk = p_game(p_return)

    @lru_cache(maxsize=None)
    def dp(a, b, a_srv):
        if a >= 6 and b <= 4 and a - b >= 2: return 1.0
        if b >= 6 and a <= 4 and b - a >= 2: return 0.0
        if a == 7: return 1.0
        if b == 7: return 0.0
        if a == 6 and b == 6:
            return p_tiebreak(p_serve, p_return) if a_srv else 1 - p_tiebreak(1 - p_return, 1 - p_serve)

        p = hold if a_srv else brk
        return p * dp(a+1, b, not a_srv) + (1-p) * dp(a, b+1, not a_srv)

    r = dp(0, 0, True)
    dp.cache_clear()
    return r


def p_set_b_starts(p_serve: float, p_return: float) -> float:
    """P(A wins set) when B serves first."""
    hold = p_game(p_serve)
    brk = p_game(p_return)

    @lru_cache(maxsize=None)
    def dp(a, b, a_srv):
        if a >= 6 and b <= 4 and a - b >= 2: return 1.0
        if b >= 6 and a <= 4 and b - a >= 2: return 0.0
        if a == 7: return 1.0
        if b == 7: return 0.0
        if a == 6 and b == 6:
            return p_tiebreak(p_serve, p_return) if a_srv else 1 - p_tiebreak(1 - p_return, 1 - p_serve)

        p = hold if a_srv else brk
        return p * dp(a+1, b, not a_srv) + (1-p) * dp(a, b+1, not a_srv)

    r = dp(0, 0, False)
    dp.cache_clear()
    return r


def p_match(p_serve: float, p_return: float, best_of: int = 3) -> float:
    """P(A wins match). Analytical."""
    sets_needed = (best_of + 1) // 2
    ps_a = p_set(p_serve, p_return)
    ps_b = p_set_b_starts(p_serve, p_return)

    @lru_cache(maxsize=None)
    def dp(a, b, a_srv):
        if a == sets_needed: return 1.0
        if b == sets_needed: return 0.0
        ps = ps_a if a_srv else ps_b
        return ps * dp(a+1, b, not a_srv) + (1-ps) * dp(a, b+1, not a_srv)

    r = dp(0, 0, True)
    dp.cache_clear()
    return r


# ─── Monte Carlo ───────────────────────────────────────────────
def sim_game(p, rng):
    a = b = 0
    while True:
        if rng.random() < p: a += 1
        else: b += 1
        if a >= 4 and a - b >= 2: return 'A'
        if b >= 4 and b - a >= 2: return 'B'


def sim_tiebreak(p_a, p_b, a_first, rng):
    a = b = 0
    for pt in range(200):
        if pt == 0: serving = a_first
        else: serving = a_first if ((pt - 1) // 2) % 2 == 0 else not a_first
        p = p_a if serving else 1 - p_b
        if rng.random() < p: a += 1
        else: b += 1
        if a >= 7 and a - b >= 2: return 'A'
        if b >= 7 and b - a >= 2: return 'B'
    return 'A'


def sim_match(p_serve_a: float, p_serve_b: float, best_of: int = 3, n: int = 20000, seed: int = 42) -> dict:
    """Monte Carlo match simulation. Returns win prob, expected games, scorelines."""
    rng = np.random.default_rng(seed)
    sets_needed = (best_of + 1) // 2

    wins = 0
    games_list, sets_list = [], []
    scores = {}

    for _ in range(n):
        a_sets = b_sets = 0
        total_games = 0
        a_serves = True
        set_scores = []

        while a_sets < sets_needed and b_sets < sets_needed:
            ag = bg = 0
            a_srv = a_serves

            while True:
                if (ag >= 6 or bg >= 6):
                    if ag >= 6 and bg <= 4 and ag - bg >= 2: break
                    if bg >= 6 and ag <= 4 and bg - ag >= 2: break
                    if ag == 7 or bg == 7: break

                if ag == 6 and bg == 6:
                    w = sim_tiebreak(p_serve_a, p_serve_b, a_srv, rng)
                    if w == 'A': ag += 1
                    else: bg += 1
                    a_srv = not a_srv
                    continue

                p = p_serve_a if a_srv else 1 - p_serve_b
                w = sim_game(p, rng)
                if w == 'A': ag += 1
                else: bg += 1
                a_srv = not a_srv

            set_scores.append((ag, bg))
            total_games += ag + bg
            if ag > bg: a_sets += 1
            else: b_sets += 1
            a_serves = not a_srv

        if a_sets > b_sets: wins += 1
        games_list.append(total_games)
        sets_list.append(a_sets + b_sets)
        key = tuple(set_scores)
        scores[key] = scores.get(key, 0) + 1

    return {
        'p_win': wins / n,
        'avg_games': np.mean(games_list),
        'std_games': np.std(games_list),
        'avg_sets': np.mean(sets_list),
        'scores': sorted(scores.items(), key=lambda x: -x[1])[:10],
        'n': n,
    }
