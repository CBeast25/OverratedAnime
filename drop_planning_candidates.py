"""
drop_planning_candidates.py

Analyze AniList "Plan to Watch" entries and rank which titles are the best
candidates to drop, defer, or keep.

This script combines your exported AniList list with your personalized
prediction output, scores each PLANNING entry for "drop pressure," and
exports two review files:
- titles most worth cutting or deprioritizing
- titles most worth keeping

Inputs:
- my_anilist_list.csv
- anilist_personalized_predictions_full.csv

Outputs:
- drop_candidates_planning.csv
- keep_candidates_planning.csv

Ranking logic:
- prioritizes low FinalScore
- also considers low predicted score
- low finish probability
- lower-confidence predictions such as Experimental

Notes:
- This script does not modify AniList
- It only produces CSVs for review and cleanup decisions
"""
import argparse
import pandas as pd
import numpy as np


DEFAULT_LIST_CSV = "my_anilist_list.csv"
DEFAULT_PRED_CSV = "anilist_personalized_predictions_full.csv"


def normalize_status(s: pd.Series) -> pd.Series:
    return s.fillna("UNKNOWN").astype(str).str.strip().str.upper()


def load_inputs(list_csv: str, pred_csv: str):
    personal = pd.read_csv(list_csv)
    pred = pd.read_csv(pred_csv)

    if "AniList ID" not in personal.columns:
        raise ValueError("List CSV must contain column 'AniList ID'.")
    if "Status" not in personal.columns:
        raise ValueError("List CSV must contain column 'Status'.")

    if "id" not in pred.columns:
        raise ValueError("Predictions CSV must contain column 'id'.")
    for c in ["Predicted_Score", "P_finish", "FinalScore", "Confidence"]:
        if c not in pred.columns:
            raise ValueError(f"Predictions CSV missing required column: {c}")

    personal["AniList ID"] = pd.to_numeric(personal["AniList ID"], errors="coerce")
    personal = personal.dropna(subset=["AniList ID"]).copy()
    personal["AniList ID"] = personal["AniList ID"].astype(int)
    personal["Status_norm"] = normalize_status(personal["Status"])

    pred["id"] = pd.to_numeric(pred["id"], errors="coerce")
    pred = pred.dropna(subset=["id"]).copy()
    pred["id"] = pred["id"].astype(int)

    for c in ["Predicted_Score", "P_finish", "FinalScore"]:
        pred[c] = pd.to_numeric(pred[c], errors="coerce")

    title_cols = [c for c in ["title_english", "title_romaji"] if c in pred.columns]
    if title_cols:
        pred["title_best"] = pred[title_cols].bfill(axis=1).iloc[:, 0]
    else:
        pred["title_best"] = pred["id"].astype(str)

    return personal, pred


def make_rank_table(personal: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    planning = personal[personal["Status_norm"].eq("PLANNING")].copy()

    if "Score" in planning.columns:
        planning["Score_num"] = pd.to_numeric(planning["Score"], errors="coerce").fillna(0)
        planning = planning.sort_values(["AniList ID", "Score_num"], ascending=[True, False])
    planning = planning.drop_duplicates("AniList ID").copy()

    merged = planning.merge(pred, left_on="AniList ID", right_on="id", how="left", suffixes=("_list", "_pred"))

    merged["Predicted_Score"] = merged["Predicted_Score"].clip(lower=1, upper=100)
    merged["P_finish"] = merged["P_finish"].clip(lower=0, upper=1)
    merged["FinalScore"] = merged["FinalScore"].clip(lower=0)

    merged["pred_score_norm"] = (merged["Predicted_Score"] - 1) / (100 - 1)
    merged["final_norm"] = merged["FinalScore"] / (merged["FinalScore"].max() if merged["FinalScore"].max() else 1.0)

    merged["DropPressure"] = (
        (1 - merged["final_norm"]) * 0.55
        + (1 - merged["P_finish"]) * 0.30
        + (1 - merged["pred_score_norm"]) * 0.15
    )

    merged["Flag_LowFinish"] = merged["P_finish"] < 0.45
    merged["Flag_LowScore"] = merged["Predicted_Score"] < 65
    merged["Flag_Experimental"] = merged["Confidence"].astype(str).str.upper().eq("EXPERIMENTAL")

    cols = [
        "AniList ID",
        "title_best",
        "Predicted_Score",
        "meanScore",
        "P_finish",
        "FinalScore",
        "Confidence",
        "DropPressure",
        "Flag_LowFinish",
        "Flag_LowScore",
        "Flag_Experimental",
    ]
    for extra in ["Progress", "Start Date", "Notes", "Genres", "Tags"]:
        if extra in merged.columns and extra not in cols:
            cols.append(extra)

    return merged[cols].sort_values(["DropPressure", "FinalScore"], ascending=[False, True]).reset_index(drop=True)


def split_drop_keep(
    df: pd.DataFrame,
    drop_top_n: int,
    drop_pressure_min: float,
    pfinish_max: float,
    predicted_score_max: float,
):
    cond = (
        (df.index < drop_top_n)
        | (df["DropPressure"] >= drop_pressure_min)
        | (df["P_finish"] <= pfinish_max)
        | (df["Predicted_Score"] <= predicted_score_max)
    )

    drop_df = df[cond].copy()
    keep_df = df[~cond].copy()

    keep_df = keep_df.sort_values(["FinalScore", "Predicted_Score"], ascending=[False, False]).reset_index(drop=True)
    return drop_df, keep_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list_csv", default=DEFAULT_LIST_CSV)
    ap.add_argument("--pred_csv", default=DEFAULT_PRED_CSV)

    ap.add_argument("--drop_top_n", type=int, default=75, help="Always include the top N highest-drop-pressure titles.")
    ap.add_argument("--drop_pressure_min", type=float, default=0.70, help="Also drop if DropPressure >= this.")
    ap.add_argument("--pfinish_max", type=float, default=0.40, help="Also drop if P_finish <= this.")
    ap.add_argument("--predicted_score_max", type=float, default=60.0, help="Also drop if Predicted_Score <= this.")

    ap.add_argument("--out_drop", default="drop_candidates_planning.csv")
    ap.add_argument("--out_keep", default="keep_candidates_planning.csv")
    args = ap.parse_args()

    personal, pred = load_inputs(args.list_csv, args.pred_csv)
    df = make_rank_table(personal, pred)

    if df.empty:
        print("No PLANNING entries found in your list export.")
        return

    drop_df, keep_df = split_drop_keep(
        df,
        drop_top_n=args.drop_top_n,
        drop_pressure_min=args.drop_pressure_min,
        pfinish_max=args.pfinish_max,
        predicted_score_max=args.predicted_score_max,
    )

    drop_df.to_csv(args.out_drop, index=False)
    keep_df.to_csv(args.out_keep, index=False)

    print(f"Planning entries analyzed: {len(df)}")
    print(f"Drop candidates: {len(drop_df)} -> {args.out_drop}")
    print(f"Keep candidates: {len(keep_df)} -> {args.out_keep}")

    print("\nDrop-candidate summary (medians):")
    print(
        drop_df[["Predicted_Score", "P_finish", "FinalScore", "DropPressure"]]
        .median(numeric_only=True)
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
