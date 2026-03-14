"""
recommend_next.py

Generate next-watch AniList recommendations from the model prediction output.

This script filters out completed or already rated entries, applies finish-score
and ranking heuristics, optionally boosts novelty or contrarian picks, and prints
a ranked recommendation table.

Inputs:
- anilist_personalized_predictions_full.csv
- my_anilist_list.csv

Outputs:
- recommendations_next_pretty.csv
- recommendations_next_raw.csv
"""
import pandas as pd
import numpy as np

ALLOW_FORMATS = {"TV"}
ALLOW_TV_SHORT = False
PRED_CSV = "anilist_personalized_predictions_full.csv"
PERSONAL_CSV = "my_anilist_list.csv" 
TOP_N = 25

MIN_P_FINISH = 0.65
MIN_PRED_SCORE = 75

ALLOW_EXPERIMENTAL = False
ONLY_UNFINISHED_ON_LIST = False

USE_SOFT_FINISH_FLOOR = True
P_FINISH_FLOOR = 0.30

USE_FRANCHISE_SATURATION = True
FRANCHISE_DECAY = 0.85

USE_CONTRARIAN_BONUS = True
CONTRARIAN_MAX_BONUS = 0.20

USE_NOVELTY_BONUS = False
NOVELTY_MAX_BONUS = 0.15

USE_FINALSCORE_CEILING = True
CEILING_MARGIN = 7.0

EXCLUDE_FORMATS = {"SPECIAL", "MUSIC"}
EXCLUDE_TV_SHORT = False

SHOW_COLUMNS = [
    "Rank", "Title", "Format",
    "Pred", "P_fin", "Final", "Mean", "Pop", "Conf"
]


def _clean_str(s):
    if pd.isna(s):
        return ""
    return str(s).strip()

def pick_title(row) -> str:
    en = _clean_str(row.get("title_english", ""))
    ro = _clean_str(row.get("title_romaji", ""))
    return en if en else ro if ro else "Unknown Title"

def pretty_table(df: pd.DataFrame, finish_col: str, top_n: int) -> pd.DataFrame:
    view = df.head(top_n).copy()

    view.insert(0, "Rank", np.arange(1, len(view) + 1))
    view["Title"] = view.apply(pick_title, axis=1)
    view["Format"] = view.get("format", "")

    view["Pred"] = pd.to_numeric(view.get("Predicted_Score", np.nan), errors="coerce")
    view["P_fin"] = pd.to_numeric(view.get(finish_col, np.nan), errors="coerce")
    view["Final"] = pd.to_numeric(view.get("FinalScore", np.nan), errors="coerce")
    view["Mean"] = pd.to_numeric(view.get("meanScore", np.nan), errors="coerce")
    view["Pop"] = pd.to_numeric(view.get("popularity", np.nan), errors="coerce")
    view["Conf"] = view.get("Confidence", "")

    view["Pred"] = view["Pred"].round(1)
    view["P_fin"] = view["P_fin"].round(3)
    view["Final"] = view["Final"].round(2)
    view["Mean"] = view["Mean"].round(0).astype("Int64")
    view["Pop"] = view["Pop"].round(0).astype("Int64")

    view = view[SHOW_COLUMNS]
    return view


def main():
    pred = pd.read_csv(PRED_CSV)
    personal = pd.read_csv(PERSONAL_CSV)

    if "id" not in pred.columns:
        raise ValueError("Predictions CSV must contain 'id' column (AniList id).")
    if "AniList ID" not in personal.columns:
        raise ValueError("Personal CSV must contain 'AniList ID' column.")
    if "Status" not in personal.columns:
        raise ValueError("Personal CSV must contain 'Status' column.")

    personal["AniList ID"] = pd.to_numeric(personal["AniList ID"], errors="coerce")
    personal = personal.dropna(subset=["AniList ID"]).copy()
    personal["AniList ID"] = personal["AniList ID"].astype(int)
    personal["Status"] = personal["Status"].fillna("UNKNOWN").astype(str).str.upper()

    personal["Score"] = pd.to_numeric(personal.get("Score", np.nan), errors="coerce")
    rated_ids = set(personal.loc[personal["Score"].fillna(0) > 0, "AniList ID"].tolist())

    completed_ids = set(personal.loc[personal["Status"] == "COMPLETED", "AniList ID"].tolist())
    all_list_ids = set(personal["AniList ID"].tolist())

    pred_ids = set(pd.to_numeric(pred["id"], errors="coerce").dropna().astype(int).tolist())

    if ONLY_UNFINISHED_ON_LIST:
        candidates = (all_list_ids - completed_ids)
    else:
        candidates = pred_ids - completed_ids

    df = pred.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)

    df = df[df["id"].isin(candidates)].copy()

    df = df[~df["id"].isin(rated_ids)].copy()

    if "title_english" not in df.columns:
        df["title_english"] = ""
    if "title_romaji" not in df.columns:
        df["title_romaji"] = ""

    if not ALLOW_EXPERIMENTAL and "Confidence" in df.columns:
        df = df[df["Confidence"].isin(["High", "Medium"])].copy()

    finish_col = None
    for c in ["P_finish_final", "P_finish_current_state", "P_finish", "P_finish_prewatch"]:
        if c in df.columns:
            finish_col = c
            break

    if finish_col is None:
        raise ValueError("No finish probability column found. Expected one of: "
                         "P_finish_final, P_finish_current_state, P_finish, P_finish_prewatch")

    df[finish_col] = pd.to_numeric(df[finish_col], errors="coerce").fillna(0.0).clip(0, 1)

    if "Predicted_Score" not in df.columns:
        raise ValueError("Predictions CSV must contain 'Predicted_Score' column.")
    df["Predicted_Score"] = pd.to_numeric(df["Predicted_Score"], errors="coerce").fillna(0.0)

    if USE_SOFT_FINISH_FLOOR:
        p_rank = np.maximum(df[finish_col].values, P_FINISH_FLOOR)
    else:
        p_rank = df[finish_col].values

    if "FinalScore" not in df.columns:
        df["FinalScore"] = df["Predicted_Score"] * p_rank
    else:
        df["FinalScore"] = pd.to_numeric(df["FinalScore"], errors="coerce").fillna(0.0)

    if USE_CONTRARIAN_BONUS and "meanScore" in df.columns:
        ms = pd.to_numeric(df["meanScore"], errors="coerce").fillna(0.0)
        ps = df["Predicted_Score"]
        contr = np.clip((ps - ms) / 20.0, 0.0, CONTRARIAN_MAX_BONUS)
        df["FinalScore"] = df["FinalScore"] * (1.0 + contr)

    if USE_NOVELTY_BONUS and "popularity" in df.columns:
        pop = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
        logp = np.log1p(np.clip(pop, 0, None))
        denom = float(logp.max()) if float(logp.max()) > 0 else 1.0
        novelty = 1.0 + NOVELTY_MAX_BONUS * (1.0 - (logp / denom))
        df["FinalScore"] = df["FinalScore"] * novelty

    if USE_FINALSCORE_CEILING:
        df["FinalScore"] = np.minimum(df["FinalScore"], df["Predicted_Score"] + CEILING_MARGIN)

    df = df.sort_values(["FinalScore", "Predicted_Score"], ascending=False).reset_index(drop=True)

    if USE_FRANCHISE_SATURATION:
        if "idMal" in df.columns:
            key = pd.to_numeric(df["idMal"], errors="coerce").fillna(0).astype(int)
            key = np.where(key.values == 0, df["id"].values, key.values)
        else:
            key = df["id"].values

        seen = {}
        penalty = np.ones(len(df), dtype=float)
        for i, k in enumerate(key):
            cnt = seen.get(int(k), 0)
            if cnt > 0:
                penalty[i] = FRANCHISE_DECAY ** cnt
            seen[int(k)] = cnt + 1

        df["FinalScore"] = df["FinalScore"] * penalty

        df = df.sort_values(["FinalScore", "Predicted_Score"], ascending=False).reset_index(drop=True)

    if "format" not in df.columns:
        raise ValueError("Predictions CSV missing 'format' column; can't filter to series-only.")

    df["format"] = df["format"].fillna("UNKNOWN").astype(str).str.upper()

    allowed = set(ALLOW_FORMATS)
    if ALLOW_TV_SHORT:
        allowed.add("TV_SHORT")

    df = df[df["format"].isin(allowed)].copy()

    MAX_EPISODES = 100
    if "episodes" in df.columns:
        ep = pd.to_numeric(df["episodes"], errors="coerce")

        known = ep.fillna(0) > 0
        df = df[known].copy()

        df = df[ep <= MAX_EPISODES].copy()

    df["RankScore"] = (
        df["FinalScore"]
        * np.clip(df[finish_col] / MIN_P_FINISH, 0.6, 1.15)
        * np.clip(df["Predicted_Score"] / MIN_PRED_SCORE, 0.7, 1.10)
    )

    if "popularity" in df.columns:
        pop = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
        logp = np.log1p(np.clip(pop, 0, None))
        denom = float(logp.max()) if float(logp.max()) > 0 else 1.0
        novelty = 1.0 + 0.10 * (1.0 - (logp / denom))
        df["RankScore"] *= novelty

    if "Confidence" in df.columns:
        surprise_map = {"High": 1.00, "Medium": 1.05, "Experimental": 1.12}
        df["RankScore"] *= df["Confidence"].map(surprise_map).fillna(1.0)

    df = df.sort_values(["RankScore", "FinalScore", "Predicted_Score"], ascending=False).reset_index(drop=True)

    TOP_SAFE = 18
    TOP_NOVEL = max(0, TOP_N - TOP_SAFE)

    if "popularity" in df.columns and TOP_NOVEL > 0:
        pop = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
        pop_cut = float(pop.quantile(0.40))

        safe = df[pop > pop_cut]
        novel = df[pop <= pop_cut]

        df = pd.concat([safe.head(TOP_SAFE), novel.head(TOP_NOVEL)], ignore_index=True)
    else:
        df = df.head(TOP_N).copy()


    top_view = pretty_table(df, finish_col=finish_col, top_n=TOP_N)

    print("\n=== Top Recommendations ===\n")
    with pd.option_context(
        "display.max_colwidth", 60,
        "display.width", 140,
        "display.colheader_justify", "left"
    ):
        print(top_view.to_string(index=False))

    if len(top_view) > 0:
        best_row = df.iloc[0]
        name = pick_title(best_row)
        print("\n=== Pick One ===")
        print(
            f"{name}  |  Pred: {float(best_row.get('Predicted_Score', np.nan)):.1f}  "
            f"|  {finish_col}: {float(best_row.get(finish_col, np.nan)):.3f}  "
            f"|  FinalScore: {float(best_row.get('FinalScore', np.nan)):.2f}"
        )

    top_view.to_csv("recommendations_next_pretty.csv", index=False)
    df.head(TOP_N).to_csv("recommendations_next_raw.csv", index=False)
    print("\nSaved: recommendations_next_pretty.csv and recommendations_next_raw.csv")


if __name__ == "__main__":
    main()