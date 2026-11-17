"""
download_data.py — Fetch ATP match data from TennisMyLife
===========================================================
TennisMyLife: https://stats.tennismylife.org
Updated daily with live ATP results. Same column format as Sackmann.
Free, MIT license.

Run this first, then run build_model.py

Usage:
    python download_data.py              # Download 2018-2026
    python download_data.py --all        # Download 1968-2026
    python download_data.py --update     # Only re-download current year
"""

import urllib.request
import os
import sys
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
API_URL = "https://stats.tennismylife.org/api/data-files"
BASE_URL = "https://stats.tennismylife.org/data"


def download_file(url: str, dest: Path) -> bool:
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def get_file_list() -> list:
    try:
        with urllib.request.urlopen(API_URL) as resp:
            data = json.loads(resp.read())
            return data.get('files', [])
    except:
        return []


def main():
    DATA_DIR.mkdir(exist_ok=True)
    args = sys.argv[1:]

    if '--all' in args:
        years = range(1968, 2027)
    elif '--update' in args:
        from datetime import datetime
        years = [datetime.now().year - 1, datetime.now().year]
    else:
        years = range(2018, 2027)

    print(f"Downloading ATP match data from TennisMyLife...")
    print(f"Years: {min(years)}-{max(years)}")
    print(f"Saving to: {DATA_DIR}\n")

    total = 0
    for year in years:
        fname = f"{year}.csv"
        url = f"{BASE_URL}/{fname}"
        dest = DATA_DIR / fname

        print(f"  {fname}...", end=" ", flush=True)
        if download_file(url, dest):
            with open(dest) as f:
                lines = sum(1 for _ in f) - 1  # minus header
            print(f"OK ({lines:,} matches)")
            total += lines
        else:
            print("SKIP")

    # Also download ongoing tournaments
    print(f"\n  ongoing_tourneys.csv...", end=" ", flush=True)
    if download_file(f"{BASE_URL}/ongoing_tourneys.csv", DATA_DIR / "ongoing_tourneys.csv"):
        print("OK")
    else:
        print("SKIP")

    print(f"\n✅ Done. {total:,} total matches downloaded to {DATA_DIR}/")
    print(f"   Next: python build_model.py")


if __name__ == "__main__":
    main()
