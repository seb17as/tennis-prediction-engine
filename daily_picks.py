"""
Tennis V2 — Daily Picks Pipeline (adapted to your project)
============================================================
Uses YOUR predict.py directly. No duplicate model loading.

Your structure:
    Tennis_V2/
    ├── predict.py          ← predict(a, b, surface, best_of)
    ├── build_model.py      ← Elo class
    ├── daily_picks.py      ← THIS FILE
    ├── data/ongoing_tourneys.csv
    └── model/ (elo.pkl, model.pkl, etc.)

Usage:
    python daily_picks.py                      # today's picks
    python daily_picks.py --date 2026-03-25    # specific date
    python daily_picks.py --dry-run            # preview only
    python daily_picks.py --from-file schedule.json  # manual schedule
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import pandas as pd
import numpy as np

# ── Setup paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
PICKS_DIR = BASE / "picks"
DATA_DIR = BASE / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("daily_picks")

# ── Import YOUR predict module ───────────────────────────────────────────────
# This loads player_features.csv, elo.pkl etc. on import
sys.path.insert(0, str(BASE))
from build_model import Elo
from predict import predict, FEATURES, ELO

# ── Odds API (optional) ─────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# ── Tier thresholds ──────────────────────────────────────────────────────────
TIERS = {
    "PREMIUM":     {"min_prob": 0.72, "min_edge": 0.08},
    "STANDARD":    {"min_prob": 0.62, "min_edge": 0.04},
    "SPECULATIVE": {"min_prob": 0.55, "min_edge": 0.00},
}


@dataclass
class Pick:
    match_id: str
    tournament: str
    surface: str
    round_name: str
    player_1: str
    player_2: str
    predicted_winner: str
    win_prob: float
    tier: str
    # Model breakdown
    elo_prob: float = 0.0
    analytical_prob: float = 0.0
    mc_prob: float = 0.0
    ensemble: float = 0.0
    # Serve/return context
    p1_serve: float = 0.0
    p2_serve: float = 0.0
    p1_eff_return: float = 0.0
    p2_eff_return: float = 0.0
    # Expected scoreline
    exp_games: float = 0.0
    exp_sets: float = 0.0
    top_scores: list = None
    # Odds & edge (if available)
    edge: float = 0.0
    odds_p1: Optional[float] = None
    odds_p2: Optional[float] = None
    scheduled_time: Optional[str] = None
    reasoning: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# 1. MATCH DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_matches(target_date: date) -> list[dict]:
    """
    Find today's matches. Priority:
      1. ongoing_tourneys.csv (your existing file) + player matching
      2. The Odds API (if key set)
      3. Manual schedule JSON
    """
    matches = []

    # ── Strategy 1: ongoing_tourneys.csv ─────────────────────────────────
    ongoing_file = DATA_DIR / "ongoing_tourneys.csv"
    if ongoing_file.exists():
        try:
            matches = _from_ongoing_tourneys(ongoing_file, target_date)
            if matches:
                log.info(f"Detected {len(matches)} matches from ongoing_tourneys.csv")
                return matches
        except Exception as e:
            log.warning(f"ongoing_tourneys parse failed: {e}")

    # ── Strategy 2: Odds API ─────────────────────────────────────────────
    if ODDS_API_KEY:
        try:
            import requests
            matches = _from_odds_api(target_date)
            if matches:
                log.info(f"Detected {len(matches)} matches from Odds API")
                return matches
        except Exception as e:
            log.warning(f"Odds API failed: {e}")

    # ── Strategy 3: Manual JSON ──────────────────────────────────────────
    manual = BASE / "schedules" / f"{target_date.isoformat()}.json"
    if manual.exists():
        with open(manual) as f:
            matches = json.load(f)
        log.info(f"Loaded {len(matches)} matches from manual schedule")
        return matches

    return matches


def _from_ongoing_tourneys(path: Path, target_date: date) -> list[dict]:
    """
    Parse ongoing_tourneys.csv for scheduled matchups.
    Adapts to whatever columns TML provides.
    """
    df = pd.read_csv(path)
    log.info(f"ongoing_tourneys.csv columns: {list(df.columns)}")

    matches = []
    for i, row in df.iterrows():
        match = {
            "match_id": f"tml_{i}",
            "tournament": row.get("tourney_name", row.get("tournament", "Unknown")),
            "surface": row.get("surface", "Hard"),
            "round_name": row.get("round", row.get("round_name", "")),
            "player_1": row.get("player_1", row.get("winner_name", "")),
            "player_2": row.get("player_2", row.get("loser_name", "")),
            "scheduled_time": row.get("time", None),
        }
        if match["player_1"] and match["player_2"]:
            matches.append(match)

    return matches


def _get_active_tennis_sports() -> list[str]:
    """Discover active tennis sport keys from The Odds API."""
    import requests
    resp = requests.get("https://api.the-odds-api.com/v4/sports",
                        params={"apiKey": ODDS_API_KEY}, timeout=15)
    resp.raise_for_status()
    sports = [s["key"] for s in resp.json()
              if "tennis" in s["key"] and s.get("active", True)]
    log.info(f"Active tennis sports: {sports}")
    return sports


def _from_odds_api(target_date: date) -> list[dict]:
    """Fetch from The Odds API — discovers active tennis tournaments dynamically."""
    import requests
    sports = _get_active_tennis_sports()
    if not sports:
        log.warning("No active tennis sports found in Odds API")
        return []

    matches = []
    for sport in sports:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
            resp = requests.get(url, params={
                "apiKey": ODDS_API_KEY,
                "regions": "us,eu",
                "markets": "h2h",
                "dateFormat": "iso",
            }, timeout=15)
            resp.raise_for_status()
            for ev in resp.json():
                commence = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
                if commence.date() != target_date:
                    continue
                matches.append({
                    "match_id": ev["id"],
                    "tournament": ev.get("sport_title", sport),
                    "surface": "Hard",
                    "round_name": "",
                    "player_1": ev["home_team"],
                    "player_2": ev["away_team"],
                    "scheduled_time": ev["commence_time"],
                })
        except Exception as e:
            log.warning(f"Odds API fetch {sport}: {e}")
    return matches


# ═══════════════════════════════════════════════════════════════════════════
# 2. ODDS FETCHING
# ═══════════════════════════════════════════════════════════════════════════

def fetch_odds(matches: list[dict]) -> dict:
    """Returns {match_id: (odds_p1, odds_p2)} decimal."""
    if not ODDS_API_KEY:
        return {}

    import requests
    odds_map = {}
    try:
        sports = _get_active_tennis_sports()
    except Exception as e:
        log.warning(f"Could not discover tennis sports for odds: {e}")
        return {}

    for sport in sports:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
            resp = requests.get(url, params={
                "apiKey": ODDS_API_KEY, "regions": "us,eu",
                "markets": "h2h", "dateFormat": "iso",
            }, timeout=15)
            resp.raise_for_status()
            for ev in resp.json():
                all_p1, all_p2 = [], []
                for bk in ev.get("bookmakers", []):
                    for mkt in bk.get("markets", []):
                        if mkt["key"] == "h2h":
                            outcomes = {o["name"]: o["price"] for o in mkt["outcomes"]}
                            if ev["home_team"] in outcomes and ev["away_team"] in outcomes:
                                all_p1.append(outcomes[ev["home_team"]])
                                all_p2.append(outcomes[ev["away_team"]])
                if all_p1:
                    odds_map[ev["id"]] = (np.mean(all_p1), np.mean(all_p2))
        except Exception as e:
            log.warning(f"Odds fetch {sport}: {e}")

    return odds_map


# ═══════════════════════════════════════════════════════════════════════════
# 3. RUN YOUR MODEL
# ═══════════════════════════════════════════════════════════════════════════

def run_prediction(p1: str, p2: str, surface: str = "Hard",
                   best_of: int = 3) -> Optional[dict]:
    """
    Calls YOUR predict() function directly.
    Returns the full prediction dict or None if player not found.
    """
    try:
        result = predict(p1, p2, surface=surface, best_of=best_of, n_sims=20000)
        return result
    except Exception as e:
        log.warning(f"Prediction failed {p1} vs {p2}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 4. TIER ASSIGNMENT & EDGE
# ═══════════════════════════════════════════════════════════════════════════

def assign_tier(prob: float, edge: float) -> str:
    for tier, thresholds in TIERS.items():
        if prob >= thresholds["min_prob"] and edge >= thresholds["min_edge"]:
            return tier
    return "NO_PICK"


def calc_edge(prob: float, decimal_odds: Optional[float]) -> float:
    if not decimal_odds or decimal_odds <= 1.0:
        return max(prob - 0.5, 0.0)
    return prob - (1.0 / decimal_odds)


def build_reasoning(pick: Pick) -> str:
    parts = []
    if pick.win_prob >= 0.70:
        parts.append(f"Strong confidence ({pick.win_prob:.0%})")
    elif pick.win_prob >= 0.62:
        parts.append(f"Solid lean ({pick.win_prob:.0%})")
    else:
        parts.append(f"Slight edge ({pick.win_prob:.0%})")

    models_agree = sum([
        pick.elo_prob > 0.5,
        pick.analytical_prob > 0.5,
        pick.mc_prob > 0.5,
    ])
    if models_agree == 3:
        parts.append("all 3 models agree")
    elif models_agree == 2:
        parts.append("2/3 models agree")

    if pick.edge > 0.10:
        parts.append("significant value vs market")
    elif pick.edge > 0.05:
        parts.append("moderate value")

    winner_serve = pick.p1_serve if pick.predicted_winner == pick.player_1 else pick.p2_serve
    if winner_serve > 0.68:
        parts.append("elite serve")

    return ". ".join(parts) + "."


# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(target_date: date, dry_run: bool = False,
                 schedule_file: Optional[str] = None) -> list[Pick]:
    log.info(f"{'[DRY RUN] ' if dry_run else ''}Picks for {target_date}")

    # ── Detect matches ───────────────────────────────────────────────────
    if schedule_file:
        with open(schedule_file) as f:
            matches = json.load(f)
        log.info(f"Loaded {len(matches)} matches from {schedule_file}")
    else:
        matches = detect_matches(target_date)

    if not matches:
        log.warning("No matches found. Create schedules/{date}.json or set ODDS_API_KEY")
        log.info('Format: [{"player_1": "Name", "player_2": "Name", '
                 '"tournament": "...", "surface": "Hard", "round_name": "R32"}]')
        return []

    log.info(f"Processing {len(matches)} matches")

    # ── Fetch odds ───────────────────────────────────────────────────────
    odds_map = fetch_odds(matches)
    log.info(f"Odds available for {len(odds_map)} matches")

    # ── Run predictions ──────────────────────────────────────────────────
    picks = []
    for m in matches:
        p1, p2 = m["player_1"], m["player_2"]
        surface = m.get("surface", "Hard")
        best_of = 5 if "Grand Slam" in m.get("tournament", "") else 3

        result = run_prediction(p1, p2, surface, best_of)
        if not result:
            log.info(f"  SKIP: {p1} vs {p2} (player not in database)")
            continue

        ensemble = result["ensemble"]
        p1_wins = ensemble >= 0.5
        winner = p1 if p1_wins else p2
        win_prob = ensemble if p1_wins else (1 - ensemble)

        # Odds & edge
        mid = m.get("match_id", f"{p1}_{p2}")
        odds_p1, odds_p2 = odds_map.get(mid, (None, None))
        winner_odds = odds_p1 if p1_wins else odds_p2
        edge = calc_edge(win_prob, winner_odds)

        # Tier
        tier = assign_tier(win_prob, edge)
        if tier == "NO_PICK":
            log.info(f"  SKIP: {p1} vs {p2} → {winner} wins ({win_prob:.1%}) | Edge: {edge:+.1%} (below threshold)")
            continue


        # Top scores
        top_scores = []
        if result.get("scores"):
            for score, count in result["scores"][:3]:
                s = " ".join([f"{g1}-{g2}" for g1, g2 in score])
                top_scores.append({"score": s, "pct": round(count / 20000 * 100, 1)})

        pick = Pick(
            match_id=mid,
            tournament=m.get("tournament", ""),
            surface=surface,
            round_name=m.get("round_name", ""),
            player_1=p1,
            player_2=p2,
            predicted_winner=winner,
            win_prob=round(win_prob, 4),
            tier=tier,
            elo_prob=round(result["elo_prob"], 4),
            analytical_prob=round(result["analytical"], 4),
            mc_prob=round(result["mc_prob"], 4),
            ensemble=round(ensemble, 4),
            p1_serve=round(result["a_serve"], 4),
            p2_serve=round(result["b_serve"], 4),
            p1_eff_return=round(result["a_eff_ret"], 4),
            p2_eff_return=round(result["b_eff_ret"], 4),
            exp_games=round(result["exp_games"], 1),
            exp_sets=round(result["exp_sets"], 2),
            top_scores=top_scores,
            edge=round(edge, 4),
            odds_p1=round(odds_p1, 2) if odds_p1 else None,
            odds_p2=round(odds_p2, 2) if odds_p2 else None,
            scheduled_time=m.get("scheduled_time"),
        )
        pick.reasoning = build_reasoning(pick)
        picks.append(pick)

    # Sort: PREMIUM first, then by win_prob descending
    tier_order = {"PREMIUM": 0, "STANDARD": 1, "SPECULATIVE": 2}
    picks.sort(key=lambda p: (tier_order.get(p.tier, 99), -p.win_prob))

    log.info(f"Generated {len(picks)} picks: "
             f"{sum(1 for p in picks if p.tier == 'PREMIUM')} PREMIUM, "
             f"{sum(1 for p in picks if p.tier == 'STANDARD')} STANDARD, "
             f"{sum(1 for p in picks if p.tier == 'SPECULATIVE')} SPECULATIVE")

    if not dry_run and picks:
        save_picks(picks, target_date)

    return picks


def save_picks(picks: list[Pick], target_date: date):
    PICKS_DIR.mkdir(parents=True, exist_ok=True)
    out = PICKS_DIR / f"{target_date.isoformat()}.json"

    payload = {
        "date": target_date.isoformat(),
        "generated_at": datetime.utcnow().isoformat(),
        "total_picks": len(picks),
        "by_tier": {
            tier: sum(1 for p in picks if p.tier == tier)
            for tier in ["PREMIUM", "STANDARD", "SPECULATIVE"]
        },
        "picks": [asdict(p) for p in picks],
    }

    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info(f"Saved → {out}")


def print_picks(picks: list[Pick]):
    if not picks:
        print("\nNo picks generated.")
        return

    print(f"\n{'='*70}")
    print(f"  TENNIS PICKS — {date.today().isoformat()}")
    print(f"  {len(picks)} picks generated")
    print(f"{'='*70}")

    emoji = {"PREMIUM": "🔥", "STANDARD": "✅", "SPECULATIVE": "🔍"}
    for p in picks:
        print(f"\n{emoji.get(p.tier, '')} [{p.tier}] {p.tournament} — {p.round_name}")
        print(f"  {p.player_1} vs {p.player_2}")
        print(f"  → {p.predicted_winner} ({p.win_prob:.1%}) | Edge: {p.edge:+.1%}")
        print(f"  Models: Elo {p.elo_prob:.1%} | Analytical {p.analytical_prob:.1%} | MC {p.mc_prob:.1%}")
        print(f"  Serve: {p.player_1} {p.p1_serve:.1%} / {p.player_2} {p.p2_serve:.1%}")
        print(f"  Expected: {p.exp_games} games, {p.exp_sets} sets")
        if p.top_scores:
            scores_str = " | ".join([f"{s['score']} ({s['pct']}%)" for s in p.top_scores])
            print(f"  Likely scores: {scores_str}")
        if p.odds_p1:
            print(f"  Odds: {p.player_1} {p.odds_p1} / {p.player_2} {p.odds_p2}")
        print(f"  {p.reasoning}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tennis V2 Daily Picks")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--from-file", type=str, default=None, help="Manual schedule JSON")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    picks = run_pipeline(target, dry_run=args.dry_run, schedule_file=args.from_file)
    print_picks(picks)
