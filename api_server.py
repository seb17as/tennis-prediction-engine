"""
Tennis V2 — API Server
=======================
FastAPI backend serving picks with tier-based access.

Endpoints:
    GET  /api/picks/today        — today's picks (filtered by tier)
    GET  /api/picks/{date}       — historical picks
    GET  /api/stats              — track record stats
    GET  /api/tiers              — pricing/tier info
    POST /api/auth/register      — user registration
    POST /api/auth/login         — login → JWT

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PICKS_DIR = BASE_DIR / "picks"
TRACK_RECORD_DIR = BASE_DIR / "track_record"

FREE_PICKS_LIMIT = 10  # Free users see 10 picks then paywall

# Tier access by subscription level
TIER_ACCESS = {
    "free": ["SPECULATIVE"],
    "starter": ["SPECULATIVE", "STANDARD"],       # $9.99/mo
    "pro": ["SPECULATIVE", "STANDARD", "PREMIUM"],  # $24.99/mo
}

PRICING = {
    "free": {"price": 0, "label": "Free", "features": [
        "10 speculative picks/day",
        "Basic match info",
        "24h delayed track record",
    ]},
    "starter": {"price": 9.99, "label": "Starter", "features": [
        "Unlimited Standard + Speculative picks",
        "Model confidence scores",
        "Full track record access",
        "Email alerts",
    ]},
    "pro": {"price": 24.99, "label": "Pro", "features": [
        "All picks including Premium",
        "Edge & value analysis",
        "Real-time alerts",
        "Surface/tournament breakdowns",
        "API access",
        "Priority support",
    ]},
}


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Tennis Picks V2",
    description="ML-powered daily tennis predictions",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ───────────────────────────────────────────────────────────────────
class PickResponse(BaseModel):
    match_id: str
    tournament: str
    surface: str
    round_name: str
    player_1: str
    player_2: str
    predicted_winner: str
    win_prob: float
    tier: str
    scheduled_time: Optional[str] = None
    # Pro-only fields
    edge: Optional[float] = None
    odds_p1: Optional[float] = None
    odds_p2: Optional[float] = None
    model_prob_p1: Optional[float] = None
    model_prob_p2: Optional[float] = None
    reasoning: Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_user_tier(api_key: Optional[str] = Query(None, alias="key")) -> str:
    """
    Determine user tier from API key.
    In production, this validates against a DB/Stripe.
    For now, uses simple key prefixes for development.
    """
    if not api_key:
        return "free"
    if api_key.startswith("pro_"):
        return "pro"
    if api_key.startswith("starter_"):
        return "starter"
    return "free"


def load_picks(target_date: str) -> dict:
    """Load picks JSON for a given date."""
    picks_file = PICKS_DIR / f"{target_date}.json"
    if not picks_file.exists():
        raise HTTPException(status_code=404, detail=f"No picks for {target_date}")
    with open(picks_file) as f:
        return json.load(f)


def filter_picks(picks: list[dict], user_tier: str, is_today: bool = True) -> list[dict]:
    """Filter picks based on user's subscription tier."""
    allowed_tiers = TIER_ACCESS.get(user_tier, ["SPECULATIVE"])
    filtered = [p for p in picks if p["tier"] in allowed_tiers]

    # Free tier: limit total picks
    if user_tier == "free" and is_today:
        filtered = filtered[:FREE_PICKS_LIMIT]

    # Strip premium fields for non-pro users
    if user_tier != "pro":
        for p in filtered:
            p.pop("edge", None)
            p.pop("odds_p1", None)
            p.pop("odds_p2", None)
            p.pop("model_prob_p1", None)
            p.pop("model_prob_p2", None)
            p.pop("reasoning", None)

    return filtered


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/picks/today")
def get_today_picks(user_tier: str = Depends(get_user_tier)):
    today = date.today().isoformat()
    data = load_picks(today)
    picks = filter_picks(data["picks"], user_tier, is_today=True)

    return {
        "date": today,
        "tier": user_tier,
        "total_available": data["total_picks"],
        "showing": len(picks),
        "picks": picks,
        "locked_count": data["total_picks"] - len(picks),
    }


@app.get("/api/picks/{target_date}")
def get_picks_by_date(target_date: str, user_tier: str = Depends(get_user_tier)):
    # Free users: only today
    if user_tier == "free" and target_date != date.today().isoformat():
        raise HTTPException(403, "Historical picks require a paid plan")

    data = load_picks(target_date)
    picks = filter_picks(data["picks"], user_tier, is_today=False)
    return {
        "date": target_date,
        "tier": user_tier,
        "total_available": data["total_picks"],
        "showing": len(picks),
        "picks": picks,
    }


@app.get("/api/stats")
def get_stats(user_tier: str = Depends(get_user_tier)):
    stats_file = TRACK_RECORD_DIR / "stats.json"
    if not stats_file.exists():
        return {"message": "No track record yet", "overall": {}, "by_tier": {}}

    with open(stats_file) as f:
        stats = json.load(f)

    # Free: overall only
    if user_tier == "free":
        return {"overall": stats.get("overall", {})}

    return stats


@app.get("/api/stats/history")
def get_stats_history(
    days: int = Query(30, ge=7, le=90),
    user_tier: str = Depends(get_user_tier),
):
    """Rolling accuracy history for charts."""
    if user_tier == "free":
        raise HTTPException(403, "Stats history requires a paid plan")

    master_file = TRACK_RECORD_DIR / "all_picks.json"
    if not master_file.exists():
        return {"history": []}

    with open(master_file) as f:
        records = json.load(f)

    # Group by date
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    recent = [r for r in records if r["date"] >= cutoff]

    from collections import defaultdict
    by_date = defaultdict(list)
    for r in recent:
        by_date[r["date"]].append(r)

    history = []
    for d in sorted(by_date.keys()):
        recs = by_date[d]
        total = len(recs)
        correct = sum(1 for r in recs if r["correct"])
        history.append({
            "date": d,
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0,
        })

    return {"history": history, "days": days}


@app.get("/api/tiers")
def get_tiers():
    return {"tiers": PRICING}


@app.get("/api/health")
def health():
    today_file = PICKS_DIR / f"{date.today().isoformat()}.json"
    return {
        "status": "ok",
        "today_picks_available": today_file.exists(),
        "timestamp": datetime.utcnow().isoformat(),
    }
