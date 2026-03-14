"""
reduce_anilist_dataset.py

Utility script to shrink the full AniList dataset into a smaller, cleaner CSV
for modeling and analysis.

What it does:
- Keeps only a curated subset of relevant columns
- Filters out low-popularity entries (default: popularity >= 1000)
- Removes MUSIC entries
- Ensures missing columns are created for consistency

Input:
- Full AniList dataset CSV

Output:
- Reduced CSV containing only selected columns and filtered rows

Usage:
    python reduce_anilist_dataset.py --in input.csv --out output.csv --pop_min 1000
"""
import argparse
import sys
import pandas as pd


KEEP_COLS = [
    "id",
    "idMal",
    "title_romaji",
    "title_english",
    "format",
    "status",
    "source",
    "season",
    "countryOfOrigin",
    "isAdult",
    "isLicensed",
    "episodes",
    "duration",
    "startDate_year",
    "endDate_year",
    "genres",
    "tags",
    "meanScore",
    "popularity",
    "favourites",
    "trending",
    "stats_scoreDistribution",
    "stats_statusDistribution",
    "coverImage_medium",
    "description",
]

def main():
    ap = argparse.ArgumentParser(description="Reduce AniList CSV to a smaller set of columns + popularity filter.")
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV path (full AniList dataset).")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV path.")
    ap.add_argument("--pop_min", type=int, default=1000, help="Minimum popularity (members) to keep. Default 1000.")
    args = ap.parse_args()

    try:
        header = pd.read_csv(args.inp, nrows=0)
    except Exception as e:
        print(f"[ERROR] Could not read input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    missing = [c for c in KEEP_COLS if c not in header.columns]
    if missing:
        print("[WARN] Missing columns in input; they will be created as empty:", missing, file=sys.stderr)

    usecols = [c for c in KEEP_COLS if c in header.columns]
    try:
        df = pd.read_csv(args.inp, usecols=usecols, low_memory=False)
    except ValueError as e:
        print(f"[WARN] usecols read failed ({e}); falling back to full CSV read.", file=sys.stderr)
        df = pd.read_csv(args.inp, low_memory=False)

    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).astype(int)
    df = df[df["popularity"] >= args.pop_min].copy()

    if "format" in df.columns:
        df["format"] = df["format"].astype(str).str.upper()
        df = df[df["format"] != "MUSIC"].copy()

    df = df[KEEP_COLS]

    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")
    print(f"Rows kept (popularity >= {args.pop_min}): {len(df):,}")

if __name__ == "__main__":
    main()
