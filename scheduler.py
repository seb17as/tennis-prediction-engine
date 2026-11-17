"""
Tennis V2 — Scheduler
======================
Runs the daily picks pipeline on a schedule.
Two modes:
  1. Standalone loop (for dev/VPS)
  2. Cron-compatible single-run (for production)

Usage:
    python scheduler.py --mode loop     # continuous loop (dev)
    python scheduler.py --mode cron     # single run (production cron)

Crontab example (run at 6 AM and 10 AM UTC daily):
    0 6,10 * * * cd /path/to/tennis_picks_pipeline && python scheduler.py --mode cron

Systemd timer alternative: see scheduler.service / scheduler.timer
"""

import argparse
import time
import json
import logging
import subprocess
from datetime import datetime, date, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scheduler")

BASE_DIR = Path(__file__).resolve().parent
PICKS_DIR = BASE_DIR / "picks"
TRACK_RECORD_DIR = BASE_DIR / "track_record"

# ── Schedule Config ──────────────────────────────────────────────────────────
# UTC times when picks should be generated
RUN_TIMES_UTC = ["06:00", "10:00", "14:00"]
# Check every N minutes in loop mode
CHECK_INTERVAL_MINUTES = 15


def should_run_now() -> bool:
    """Check if current UTC time matches a scheduled run time."""
    now = datetime.utcnow().strftime("%H:%M")
    return now in RUN_TIMES_UTC


def picks_exist_today() -> bool:
    """Check if today's picks have already been generated."""
    today_file = PICKS_DIR / f"{date.today().isoformat()}.json"
    return today_file.exists()


def run_picks_pipeline():
    """Execute the daily picks pipeline."""
    log.info("Starting daily picks pipeline...")
    try:
        result = subprocess.run(
            ["python", str(BASE_DIR / "daily_picks.py")],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )
        if result.returncode == 0:
            log.info("Pipeline completed successfully")
            log.info(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        else:
            log.error(f"Pipeline failed (code {result.returncode})")
            log.error(result.stderr[-500:])
    except subprocess.TimeoutExpired:
        log.error("Pipeline timed out after 5 minutes")
    except Exception as e:
        log.error(f"Pipeline error: {e}")


def update_track_record():
    """
    Check yesterday's picks against actual results.
    Updates the track record for the dashboard.
    """
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    picks_file = PICKS_DIR / f"{yesterday}.json"

    if not picks_file.exists():
        return

    try:
        with open(picks_file) as f:
            picks_data = json.load(f)

        # TODO: Fetch actual results from TML or flashscore
        # For now, results are updated manually or via a separate script
        results_file = BASE_DIR / "results" / f"{yesterday}.json"
        if not results_file.exists():
            log.info(f"No results file for {yesterday} yet")
            return

        with open(results_file) as f:
            results = json.load(f)

        # Match picks to results
        TRACK_RECORD_DIR.mkdir(parents=True, exist_ok=True)

        record = []
        for pick in picks_data["picks"]:
            mid = pick["match_id"]
            if mid in results:
                actual_winner = results[mid]["winner"]
                correct = actual_winner == pick["predicted_winner"]
                record.append({
                    "date": yesterday,
                    "match_id": mid,
                    "tournament": pick["tournament"],
                    "player_1": pick["player_1"],
                    "player_2": pick["player_2"],
                    "predicted_winner": pick["predicted_winner"],
                    "actual_winner": actual_winner,
                    "win_prob": pick["win_prob"],
                    "edge": pick["edge"],
                    "tier": pick["tier"],
                    "correct": correct,
                })

        # Append to master track record
        master_file = TRACK_RECORD_DIR / "all_picks.json"
        existing = []
        if master_file.exists():
            with open(master_file) as f:
                existing = json.load(f)

        existing.extend(record)
        with open(master_file, "w") as f:
            json.dump(existing, f, indent=2)

        correct_count = sum(1 for r in record if r["correct"])
        log.info(f"Track record updated: {correct_count}/{len(record)} correct for {yesterday}")

    except Exception as e:
        log.error(f"Track record update failed: {e}")


def compute_stats():
    """Compute aggregate stats for the dashboard."""
    master_file = TRACK_RECORD_DIR / "all_picks.json"
    if not master_file.exists():
        return {}

    with open(master_file) as f:
        records = json.load(f)

    if not records:
        return {}

    stats = {"overall": {}, "by_tier": {}}

    # Overall
    total = len(records)
    correct = sum(1 for r in records if r["correct"])
    stats["overall"] = {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0,
        "roi": _compute_roi(records),
    }

    # By tier
    for tier in ["PREMIUM", "STANDARD", "SPECULATIVE"]:
        tier_recs = [r for r in records if r["tier"] == tier]
        t_total = len(tier_recs)
        t_correct = sum(1 for r in tier_recs if r["correct"])
        stats["by_tier"][tier] = {
            "total": t_total,
            "correct": t_correct,
            "accuracy": round(t_correct / t_total, 4) if t_total else 0,
            "roi": _compute_roi(tier_recs),
        }

    stats_file = TRACK_RECORD_DIR / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def _compute_roi(records: list) -> float:
    """Simplified flat-bet ROI calculation."""
    if not records:
        return 0.0
    # Assume 1 unit bet at implied odds from edge
    profit = 0.0
    for r in records:
        if r["correct"]:
            # Approximate payout: 1/prob - 1 unit wagered
            prob = r.get("win_prob", 0.5)
            payout = (1.0 / prob) - 1.0 if prob > 0 else 0
            profit += payout
        else:
            profit -= 1.0
    return round(profit / len(records), 4)


# ── Main Loop ────────────────────────────────────────────────────────────────

def loop_mode():
    """Continuous scheduler for dev/VPS deployment."""
    log.info("Scheduler started in LOOP mode")
    log.info(f"Scheduled run times (UTC): {RUN_TIMES_UTC}")
    log.info(f"Check interval: {CHECK_INTERVAL_MINUTES} min")

    while True:
        try:
            if should_run_now() and not picks_exist_today():
                run_picks_pipeline()
                update_track_record()
                compute_stats()

            time.sleep(CHECK_INTERVAL_MINUTES * 60)

        except KeyboardInterrupt:
            log.info("Scheduler stopped by user")
            break
        except Exception as e:
            log.error(f"Scheduler error: {e}")
            time.sleep(60)


def cron_mode():
    """Single-run mode for cron jobs."""
    log.info("Scheduler running in CRON mode (single execution)")

    if picks_exist_today():
        log.info("Today's picks already exist. Updating track record only.")
    else:
        run_picks_pipeline()

    update_track_record()
    compute_stats()
    log.info("Cron run complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tennis V2 Scheduler")
    parser.add_argument("--mode", choices=["loop", "cron"], default="cron")
    args = parser.parse_args()

    if args.mode == "loop":
        loop_mode()
    else:
        cron_mode()
