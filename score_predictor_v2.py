"""
score_predictor_v2.py

Train two personalized AniList models from your exported list plus the full
AniList dataset:

1. A score model that predicts your personal rating.
2. A finish model that estimates whether you are likely to finish a title.

The finish model treats:
- COMPLETED entries as positive examples
- DROPPED entries as negative examples
- CURRENT / PAUSED / ON_HOLD entries as inferred negatives only when they
  appear stale relative to your learned completion pace

Inputs:
- my_anilist_list.csv
- anilist_anime_data_complete.csv

Output:
- anilist_personalized_predictions_full.csv

Notes:
- Uses AniList genres, tags, metadata, and community-distribution fields
- Uses XGBoost for both score and finish prediction
- Uses tqdm for batched prediction and explanation passes
"""

import json
from collections import Counter

import numpy as np
import pandas as pd
import shap

from tqdm import tqdm
from scipy import sparse
from scipy.stats import spearmanr, pearsonr

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier, XGBRegressor
from xgboost.callback import EarlyStopping

PERSONAL_CSV = "my_anilist_list.csv"
ANILIST_CSV = "anilist_anime_data_complete.csv"
OUT_CSV = "anilist_personalized_predictions_full.csv"

ONLY_COMPLETED_FOR_SCORE_TRAIN = True
USE_COMMUNITY_CONTEXT = True
BATCH_SIZE = 1024

MIN_FREQ_GENRE = 3
MIN_FREQ_TAG_SCORE = 5
MIN_FREQ_TAG_FINISH = 10


FINISH_MIN_DAYS_FLOOR = 30
FINISH_MULTIPLIER = 1.6
FINISH_MIN_PROGRESS_FRACTION = 0.15


SCORE_SPLIT_MODE = "time" 
FINISH_SPLIT_MODE = "time" 
TEST_FRACTION = 0.20

FINAL_SCORE_FLOOR = 0.0

status_map = {
    "CURRENT": "current",
    "COMPLETED": "completed",
    "PLANNING": "planning",
    "DROPPED": "dropped",
    "PAUSED": "paused",
    "ON_HOLD": "paused",
}

def safe_json_loads(x, default):
    """
    Parse a JSON-like value safely.

    Args:
        x: Value that may already be a Python object, a JSON string, or missing.
        default: Value to return when parsing fails or the input is empty.

    Returns:
        The parsed Python object when possible, otherwise `default`.

    Notes:
        - Lists and dicts are returned unchanged.
        - Missing values and invalid JSON fall back to `default`.
    """
    if pd.isna(x):
        return default
    if isinstance(x, (list, dict)):
        return x
    s = str(x).strip()
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def parse_genres_series(series: pd.Series) -> list[list[str]]:
    """
    Parse AniList genre values into a list of string labels per row.

    Args:
        series: Pandas Series containing JSON arrays or already-materialized lists.

    Returns:
        A list where each element is a list of genre names for the corresponding row.
    """
    out = []
    for v in series:
        arr = safe_json_loads(v, default=[])
        out.append([str(x) for x in arr] if isinstance(arr, list) else [])
    return out


def parse_tags_series(series: pd.Series) -> list[list[str]]:
    """
    Parse AniList tag payloads into tag-name lists.

    Args:
        series: Pandas Series containing JSON arrays of tag objects.

    Returns:
        A list where each element is a list of tag names for the corresponding row.

    Notes:
        Only the `name` field is kept from each tag object.
    """
    out = []
    for v in series:
        arr = safe_json_loads(v, default=[])
        if isinstance(arr, list):
            names = []
            for item in arr:
                if isinstance(item, dict) and "name" in item:
                    names.append(str(item["name"]))
            out.append(names)
        else:
            out.append([])
    return out


def multilabel_to_indicator_frame(list_of_lists, prefix: str, min_freq: int) -> pd.DataFrame:
    """
    Convert multilabel string lists into a binary indicator dataframe.

    Args:
        list_of_lists: Sequence of label lists, one per row.
        prefix: Prefix applied to each output column name.
        min_freq: Minimum document frequency required for a label to be kept.

    Returns:
        A dataframe of 0/1 indicator columns for labels that meet `min_freq`.

    Notes:
        Duplicate labels within the same row are counted once.
    """
    c = Counter()
    for lst in list_of_lists:
        c.update(set(lst))
    keep = sorted([k for k, v in c.items() if v >= min_freq])

    data = np.zeros((len(list_of_lists), len(keep)), dtype=np.uint8)
    idx = {k: j for j, k in enumerate(keep)}
    for i, lst in enumerate(list_of_lists):
        for k in set(lst):
            j = idx.get(k)
            if j is not None:
                data[i, j] = 1

    cols = [f"{prefix}{k}" for k in keep]
    return pd.DataFrame(data, columns=cols)


def align_indicator_columns(full_lists, template_cols, prefix):
    """
    Build indicator columns using an existing training-time label vocabulary.

    Args:
        full_lists: Sequence of label lists to encode.
        template_cols: Output columns to match exactly.
        prefix: Prefix used in `template_cols`.

    Returns:
        A dataframe with the same columns and ordering as `template_cols`.

    Notes:
        Labels not present in the template are ignored.
    """
    labels = [c[len(prefix):] for c in template_cols]
    idx = {lab: j for j, lab in enumerate(labels)}
    data = np.zeros((len(full_lists), len(labels)), dtype=np.uint8)

    for i, lst in enumerate(full_lists):
        for k in set(lst):
            j = idx.get(k)
            if j is not None:
                data[i, j] = 1

    return pd.DataFrame(data, columns=template_cols)

def year_bin(y: pd.Series) -> pd.Series:
    """
    Bucket release years into coarse time periods.

    Args:
        y: Series of numeric or parseable year values.

    Returns:
        A series of year-bin labels such as `1990s`, `2015_2019`, or `UNK`.
    """
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    out = pd.Series("UNK", index=y.index, dtype=object)

    bins = [
        (1, 1989, "pre_1990"),
        (1990, 1999, "1990s"),
        (2000, 2009, "2000s"),
        (2010, 2014, "2010_2014"),
        (2015, 2019, "2015_2019"),
        (2020, 2021, "2020_2021"),
        (2022, 2023, "2022_2023"),
        (2024, 2100, "2024_plus"),
    ]

    for lo, hi, name in bins:
        m = (y >= lo) & (y <= hi)
        out.loc[m] = name

    return out

def _pick_present_signals(
    row: pd.Series,
    importance_df: pd.DataFrame,
    top_n: int = 30,
    max_hits: int = 6,
) -> list[str]:
    if importance_df is None or len(importance_df) == 0:
        return []

    candidates = importance_df["feature"].head(top_n).tolist()

    hits = []
    counts = {
        "prior": 0,
        "stats": 0,
        "content": 0,
        "meta": 0,
        "other": 0,
    }

    caps = {
        "prior": 1,
        "stats": 2,
        "content": 4,
        "meta": 2,
        "other": 2,
    }

    for f in candidates:
        if f in EXPLANATION_BLACKLIST:
            continue
        if f not in row.index:
            continue

        v = row[f]

        present = False
        if isinstance(v, (int, float, np.integer, np.floating)):
            if f.startswith(("tag_", "genre_")):
                present = int(v) == 1
            else:
                present = abs(float(v)) > 1e-9
        else:
            s = str(v).strip().upper()
            present = bool(s and s != "UNKNOWN")

        if not present:
            continue

        t = _signal_type(f)
        if counts[t] >= caps[t]:
            continue

        hits.append(f)
        counts[t] += 1

        if len(hits) >= max_hits:
            break

    return hits

GLOBAL_PRIORS = {
    "log1p_popularity",
    "startDate_year",
    "endDate_year",
    "duration",
    "episodes",
    "meanScore_10",
    
}

EXPLANATION_BLACKLIST = {
    "startDate_year",
    "endDate_year",
}

def _signal_type(name: str) -> str:
    if name.startswith(("tag_", "genre_")):
        return "content"
    if name.startswith(("format_", "source_", "season_", "countryOfOrigin_")):
        return "meta"
    if name.startswith("stats_"):
        return "stats"
    if name in GLOBAL_PRIORS:
        return "prior"
    return "other"

def _pretty_signal(sig: str) -> str:
    if sig.startswith("tag_"):
        return f"Tag: {sig[len('tag_'):]}"
    if sig.startswith("genre_"):
        return f"Genre: {sig[len('genre_'):]}"
    return sig


def make_user_score_quantiles(personal: pd.DataFrame):
    """
    Compute score quantiles from the user's historical ratings.

    Args:
        personal: Personal list dataframe containing a `Score10` column.

    Returns:
        A 5-tuple of quantile cutoffs: (q20, q40, q60, q80, q95).

    Notes:
        Falls back to fixed thresholds when too few scored entries are available.
    """
    s = personal.loc[personal["Score10"].notna(), "Score10"].astype(float).values
    if len(s) < 20:
        return (5.0, 6.0, 7.0, 8.0, 9.0)

    q20, q40, q60, q80, q95 = np.percentile(s, [20, 40, 60, 80, 95])
    return float(q20), float(q40), float(q60), float(q80), float(q95)


def _score_bucket_quantile(pred10: float, q20: float, q40: float, q60: float, q80: float, q95: float) -> str:
    if pred10 >= q95:
        return "Top-tier"
    if pred10 >= q80:
        return "Strong"
    if pred10 >= q60:
        return "Good"
    if pred10 >= q40:
        return "Mid"
    if pred10 >= q20:
        return "Weak"
    return "Skip-tier"

def build_why_score(
    i: int,
    out_row: pd.Series,
    X_all_score_row: pd.Series,
    score_imp_df: pd.DataFrame | None,
    q20: float, q40: float, q60: float, q80: float, q95: float,
):
    """
    Build a short human-readable score summary for one prediction.

    Args:
        i: Row index in the full prediction set.
        out_row: Output row containing the predicted score fields.
        X_all_score_row: Feature row for the same anime.
        score_imp_df: Optional score-importance dataframe.
        q20, q40, q60, q80, q95: User-specific score quantile cutoffs.

    Returns:
        A qualitative bucket label such as `Top-tier`, `Strong`, or `Skip-tier`.

    Notes:
        The current implementation returns only a score bucket, but the signature
        leaves room for richer explanation logic later.
    """
    pred100 = float(out_row.get("Predicted_Score", 0) or 0)
    pred10 = pred100 / 10.0

    bucket = _score_bucket_quantile(pred10, q20, q40, q60, q80, q95)
    return bucket

def build_why_finish(
    i: int,
    out_row: pd.Series,
    X_all_finish_row: pd.Series,
    finish_imp_df: pd.DataFrame | None,
    finish_threshold: float,
) -> str:
    """
    Build a short human-readable finish-likelihood summary for one prediction.

    Args:
        i: Row index in the full prediction set.
        out_row: Output row containing finish-probability fields.
        X_all_finish_row: Feature row for the same anime.
        finish_imp_df: Optional finish-importance dataframe.
        finish_threshold: Probability threshold used for the finish label.

    Returns:
        A qualitative finish summary such as `Likely to finish` or
        `Risky (may not finish)`.
    """
    p = float(out_row.get("P_finish", 0) or 0)

    if p >= max(0.85, finish_threshold):
        bucket = "Very likely to finish"
    elif p >= finish_threshold:
        bucket = "Likely to finish"
    elif p >= 0.50:
        bucket = "Uncertain"
    else:
        bucket = "Risky (may not finish)"

    bits = [f"{bucket}"]

    why = f"{', '.join(bits)}"
    return why


def is_valid_longform_anime(df: pd.DataFrame) -> pd.Series:
    """
    True for entries that behave like episodic anime or movies.
    Excludes:
      - MUSIC
      - single-episode OVA / SPECIAL
    """
    fmt = df.get("format", "").astype(str).str.upper()
    eps = pd.to_numeric(df.get("episodes", 0), errors="coerce").fillna(0)

    is_music = (fmt == "MUSIC")
    is_single_ova = (eps <= 1) & (fmt.isin(["OVA", "SPECIAL"]))

    return ~(is_music | is_single_ova)

def _get_ohe_feature_names(ohe, input_features):
    try:
        return ohe.get_feature_names_out(input_features)
    except Exception:
        names = []
        cats = ohe.categories_
        for f, cat_list in zip(input_features, cats):
            for c in cat_list:
                names.append(f"{f}={c}")
        return np.array(names, dtype=object)

def get_preprocessed_feature_names(prep: ColumnTransformer, X_sample: pd.DataFrame):
    """
    Recover output feature names from a fitted ColumnTransformer.

    Args:
        prep: Fitted ColumnTransformer.
        X_sample: Sample input dataframe used only to resolve column names.

    Returns:
        A NumPy array of transformed feature names in output order.

    Notes:
        Supports the transformer layout used in this file, including categorical
        one-hot encoded features and passthrough dense/binary features.
    """
    names = []

    for name, transformer, cols in prep.transformers_:
        if name == "remainder" and transformer == "drop":
            continue

        if hasattr(cols, "__iter__") and not isinstance(cols, (str, bytes)):
            cols_list = list(cols)
        else:
            cols_list = [cols]

        if name == "cat":
            ohe = transformer
            ohe_names = _get_ohe_feature_names(ohe, cols_list)
            names.extend(list(ohe_names))
        elif name == "num":
            names.extend(cols_list)
        else:
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    out = transformer.get_feature_names_out(cols_list)
                    names.extend(list(out))
                except Exception:
                    names.extend([f"{name}__{c}" for c in cols_list])
            else:
                names.extend([f"{name}__{c}" for c in cols_list])

    return np.array(names, dtype=object)

def export_permutation_importance(
    pipeline,
    X: pd.DataFrame,
    y,
    out_csv: str,
    top_k: int = 60,
    title: str = "",
    n_repeats: int = 10,
    max_rows: int = 5000,
    random_state: int = 42,
    scoring: str | None = None,
):
    """
    Compute permutation importance on original input columns and export the result.

    Args:
        pipeline: Fitted sklearn-style estimator or wrapper.
        X: Evaluation features in pre-transformed dataframe form.
        y: Evaluation target aligned with `X`.
        out_csv: Destination CSV path.
        top_k: Number of top features to print.
        title: Optional label for console output.
        n_repeats: Number of random permutations per feature.
        max_rows: Maximum number of evaluation rows to use.
        random_state: Seed used for sampling and permutation.
        scoring: Optional sklearn scoring name.

    Returns:
        A dataframe sorted by descending mean importance.
    """
    if len(X) > max_rows:
        X_use = X.sample(n=max_rows, random_state=random_state)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_use = y.loc[X_use.index]
        else:
            y_use = np.asarray(y)[X_use.index]
    else:
        X_use = X
        y_use = y

    result = permutation_importance(
        pipeline,
        X_use,
        y_use,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring=scoring
    )

    df_imp = pd.DataFrame({
        "feature": X_use.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    df_imp.to_csv(out_csv, index=False)

    if title:
        print(f"\nTop {top_k} permutation importances for {title}:")
    else:
        print(f"\nTop {top_k} permutation importances:")
    with pd.option_context("display.float_format", "{:.8f}".format):
        print(df_imp.head(top_k).to_string(index=False))


    print(f"\nSaved: {out_csv}")
    return df_imp

def parse_progress_col(series: pd.Series) -> pd.Series:
    """
    Normalize the personal-list progress column to integer episode counts.

    Args:
        series: Raw progress values from the exported personal list.

    Returns:
        A numeric series with invalid or missing values coerced to 0.
    """
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "N/A": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def safe_start_date(series: pd.Series) -> pd.Series:
    """
    Parse date strings safely, treating AniList zero dates as missing.

    Args:
        series: Raw date-like series.

    Returns:
        A pandas datetime series with invalid dates coerced to NaT.
    """
    s = series.replace("0000-00-00", pd.NA)
    return pd.to_datetime(s, errors="coerce")


def parse_date_to_year(series: pd.Series) -> pd.Series:
    """
    Extract year values from a date-like series.

    Args:
        series: Raw date-like series.

    Returns:
        An integer year series with missing dates filled as 0.
    """
    dt = safe_start_date(series)
    return dt.dt.year.fillna(0).astype(int)


def build_days_per_episode_by_bucket(completed_df: pd.DataFrame) -> dict:
    """
    Estimate the user's watch pace by episode-count bucket.

    Args:
        completed_df: Completed entries containing `StartDT`, `CompletedDT`,
            and `episodes`.

    Returns:
        A mapping from episode bucket (`short`, `mid`, `long`, `very_long`)
        to median days per episode.

    Notes:
        The estimate is cleaned with multiple sanity filters to avoid extreme
        completion times distorting the pace model.
    """
    tmp = completed_df.copy()
    tmp = tmp.dropna(subset=["StartDT", "CompletedDT"])
    tmp["days"] = (tmp["CompletedDT"] - tmp["StartDT"]).dt.days.clip(lower=1)

    tmp["episodes_used"] = pd.to_numeric(tmp.get("episodes", 0), errors="coerce").fillna(0)
    tmp = tmp[tmp["episodes_used"] > 0].copy()

    tmp["dpe"] = (tmp["days"] / tmp["episodes_used"]).replace([np.inf, -np.inf], np.nan)
    tmp = tmp.dropna(subset=["dpe"])
    tmp = tmp[(tmp["dpe"] >= 0) & (tmp["dpe"] <= 30)]

    def bucket(ep):
        ep = int(ep)
        if ep <= 12:
            return "short"
        if ep <= 26:
            return "mid"
        if ep <= 60:
            return "long"
        return "very_long"

    tmp["bucket"] = tmp["episodes_used"].astype(int).apply(bucket)
    min_days_by_bucket = {
        "short": 1,
        "mid": 2,
        "long": 7,
        "very_long": 14,
    }
    mins = tmp["bucket"].map(min_days_by_bucket).fillna(1).astype(int)
    tmp = tmp[tmp["days"] >= mins].copy()

    tmp["dpe"] = (tmp["days"] / tmp["episodes_used"]).replace([np.inf, -np.inf], np.nan)
    tmp = tmp.dropna(subset=["dpe"])

    tmp = tmp[(tmp["dpe"] >= 0.25) & (tmp["dpe"] <= 30)]

    med = tmp.groupby("bucket")["dpe"].median().to_dict()
    overall = float(tmp["dpe"].median()) if len(tmp) else 2.0

    for b in ["short", "mid", "long", "very_long"]:
        if b not in med or np.isnan(med[b]):
            med[b] = overall

    for b in med:
        if np.isnan(med[b]):
            med[b] = 2.0

    return med


def infer_finish_label(
    row,
    today: pd.Timestamp,
    dpe_by_bucket: dict,
    min_days_floor: int,
    multiplier: float,
    min_progress_fraction: float
):
    """
    Infer a supervised finish/drop label from personal-list state.

    Args:
        row: Row containing user status, progress, anime status, episode count,
            and start date.
        today: Reference date used to measure staleness.
        dpe_by_bucket: Learned days-per-episode pace by episode bucket.
        min_days_floor: Minimum stale threshold in days.
        multiplier: Multiplier applied to expected completion time to define
            the stale cutoff.
        min_progress_fraction: Early-progress protection threshold.

    Returns:
        1 if the title should count as finished,
        0 if it should count as dropped or effectively dropped,
        None if the case should remain unresolved and excluded from training.

    Notes:
        - Explicit COMPLETED and DROPPED statuses are respected.
        - PLANNING entries remain unresolved.
        - CURRENT entries for actively releasing titles are not inferred dropped.
        - In-progress entries are treated as dropped only when they appear stale
          relative to the user's learned completion pace.
    """
    status = str(row.get("Status", "UNKNOWN")).upper()
    episodes = pd.to_numeric(row.get("episodes", 0), errors="coerce")
    episodes = 0 if pd.isna(episodes) else int(episodes)

    progress = pd.to_numeric(row.get("Progress_int", 0), errors="coerce")
    progress = 0 if pd.isna(progress) else int(progress)

    if status == "COMPLETED":
        return 1
    if status == "DROPPED":
        return 0
    if status == "PLANNING":
        return None

    if episodes > 0 and progress >= episodes:
        return 1

    if status not in {"CURRENT", "PAUSED", "ON_HOLD"}:
        return None

    anime_status = str(row.get("status", "UNKNOWN")).upper()
    if status == "CURRENT" and anime_status in {"RELEASING", "NOT_YET_RELEASED"}:
        return None

    start_dt = row.get("StartDT", pd.NaT)
    if pd.isna(start_dt):
        return None

    days_since_start = int((today - start_dt).days)
    if days_since_start < 0:
        return None

    if episodes > 0:
        frac = progress / max(episodes, 1)
        if frac < min_progress_fraction:
            if days_since_start < max(60, min_days_floor):
                return None

    def bucket(ep: int) -> str:
        if ep <= 12:
            return "short"
        if ep <= 26:
            return "mid"
        if ep <= 60:
            return "long"
        return "very_long"

    ep_for_calc = episodes if episodes > 0 else 12
    b = bucket(ep_for_calc)

    dpe = float(dpe_by_bucket.get(b, dpe_by_bucket.get("short", 2.0))) if dpe_by_bucket else 2.0
    expected_finish_days = max(1.0, ep_for_calc * dpe)

    cutoff_days = int(max(min_days_floor, multiplier * expected_finish_days))

    if days_since_start >= cutoff_days:
        return 0

    return None

def parse_status_score_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand AniList community status and score distributions into numeric features.

    Args:
        df: AniList dataframe containing `stats_statusDistribution` and
            `stats_scoreDistribution` JSON columns.

    Returns:
        A dataframe of engineered status- and score-distribution features.

    Notes:
        The output includes totals, proportions, retention-style features,
        entropy, and tail-mass features such as `stats_p90plus`.
    """
    out = pd.DataFrame(index=df.index)

    def status_feats(x):
        arr = safe_json_loads(x, default=[])
        d = {status_map.get(i.get("status"), str(i.get("status")).lower()): int(i.get("amount", 0))
             for i in arr if isinstance(i, dict)}
        n_completed = d.get("completed", 0)
        n_dropped = d.get("dropped", 0)
        n_current = d.get("current", 0)
        n_planning = d.get("planning", 0)
        n_paused = d.get("paused", 0)
        n_total = n_completed + n_dropped + n_current + n_planning + n_paused

        if n_total <= 0:
            return (0,0,0,0,0,0, 0,0,0,0, 0,0,0)

        p_completed = n_completed / n_total
        p_dropped = n_dropped / n_total
        p_inprog = (n_current + n_paused) / n_total
        p_planning = n_planning / n_total

        completion_over_drop = n_completed / (n_dropped + 1.0)
        retention = n_completed / (n_completed + n_dropped + n_paused + 1.0)
        hype_gap = p_planning - p_completed

        return (n_total, n_completed, n_dropped, n_current, n_paused, n_planning,
                p_completed, p_dropped, p_inprog, p_planning,
                completion_over_drop, retention, hype_gap)

    cols = [
        "stats_n_total","stats_n_completed","stats_n_dropped","stats_n_current","stats_n_paused","stats_n_planning",
        "stats_p_completed","stats_p_dropped","stats_p_inprogress","stats_p_planning",
        "stats_completion_over_drop","stats_retention","stats_hype_gap"
    ]

    vals = df.get("stats_statusDistribution", pd.Series([None]*len(df))).apply(status_feats).tolist()
    out[cols] = pd.DataFrame(vals, index=df.index)
    out["stats_log1p_n_total"] = np.log1p(out["stats_n_total"].clip(lower=0))
    out["stats_has_status"] = (out["stats_n_total"] > 0).astype(int)

    def score_feats(x):
        arr = safe_json_loads(x, default=[])
        pairs = [(int(i.get("score", 0)), int(i.get("amount", 0))) for i in arr if isinstance(i, dict)]
        total = sum(a for _, a in pairs)
        if total <= 0:
            return (0,0,0,0,0,0)

        xs = np.array([s/10.0 for s,_ in pairs], dtype=float)
        ws = np.array([a for _,a in pairs], dtype=float) / total

        mean = float((xs * ws).sum())
        var = float(((xs - mean)**2 * ws).sum())
        std = float(np.sqrt(max(var, 0.0)))

        eps = 1e-12
        ent = float(-(ws * np.log(ws + eps)).sum())

        p90 = float(ws[xs >= 9].sum())
        p80 = float(ws[xs >= 8].sum())
        p50m = float(ws[xs <= 5].sum())

        return (mean, std, ent, p90, p80, p50m)

    sc_cols = ["stats_score_mean","stats_score_std","stats_score_entropy","stats_p90plus","stats_p80plus","stats_p50minus"]
    sc_vals = df.get("stats_scoreDistribution", pd.Series([None]*len(df))).apply(score_feats).tolist()
    out[sc_cols] = pd.DataFrame(sc_vals, index=df.index)
    out["stats_has_score"] = (out["stats_score_mean"] > 0).astype(int)

    return out

def report_single_anime_score(
    anime_id: int,
    anilist_valid: pd.DataFrame,
    anilist_pos: pd.Series,
    X_all_score: pd.DataFrame,
    prep_score,
    xgb_score,
    shap_values,
    score_feature_names,
    resid_bias: float,
    mean_score_10_all: np.ndarray,
    top_k: int = 30,
):
    """
    Print a SHAP-based explanation for one anime's score prediction.

    Args:
        anime_id: AniList ID to explain.
        anilist_valid: Filtered AniList dataframe aligned with feature rows.
        anilist_pos: Mapping from AniList ID to row index.
        X_all_score: Raw score feature matrix.
        prep_score: Fitted score preprocessor.
        xgb_score: Fitted score model.
        shap_values: Precomputed SHAP values aligned to transformed rows.
        score_feature_names: Names of transformed score features.
        resid_bias: Residual-bias correction applied after prediction.
        mean_score_10_all: Crowd mean score on the 1–10 scale.
        top_k: Number of top feature contributions to print.

    Returns:
        None. Prints a formatted explanation to stdout.
    """
    idx = anilist_pos.get(int(anime_id), None)
    if idx is None:
        print(f"[ERROR] anime_id {anime_id} not found in anilist_valid.")
        return

    title = anilist_valid.loc[idx, "title_romaji"] if "title_romaji" in anilist_valid.columns else ""
    print("=" * 80)
    print(f"Score explanation for ID={anime_id}  Title={title}")
    print("=" * 80)

    X_row = X_all_score.iloc[[idx]]
    X_row_t = prep_score.transform(X_row)

    pred_resid = float(xgb_score.predict(X_row_t)[0] - resid_bias)
    crowd = float(mean_score_10_all[idx])
    pred10 = float(np.clip(crowd + pred_resid, 1.0, 10.0))
    pred100 = float(np.round(pred10 * 10.0, 1))

    print(f"Crowd meanScore (1–10): {crowd:.3f}")
    print(f"Predicted residual (you - crowd): {pred_resid:+.3f}")
    print(f"Predicted your score (1–10): {pred10:.3f}")
    print(f"Predicted your score (1–100): {pred100:.1f}")
    print()

    v = shap_values[idx]
    try:
        explainer = shap.TreeExplainer(xgb_score)
        base = float(explainer.expected_value)
    except Exception:
        base = float(np.mean(xgb_score.predict(prep_score.transform(X_all_score.iloc[:2000]))))

    def _clean(n: str) -> str:
        n = n.replace("cat__", "").replace("dense__", "").replace("bin__", "")
        return n

    fnames = np.array([_clean(str(n)) for n in score_feature_names], dtype=object)

    pred_from_shap = base + float(np.sum(v))

    print(f"SHAP base value: {base:+.4f}")
    print(f"Base + sum(SHAP): {pred_from_shap:+.4f}   (should ~ predicted residual before bias correction)")
    print()

    df_rep = pd.DataFrame({
        "feature": fnames,
        "shap": v,
        "abs_shap": np.abs(v),
    }).sort_values("abs_shap", ascending=False)

    try:
        xvals = np.asarray(X_row_t.todense()).ravel()
    except Exception:
        xvals = np.asarray(X_row_t).ravel()

    df_rep["value_transformed"] = xvals[df_rep.index.values]

    print(f"Top {top_k} feature contributions (positive pushes score up, negative down):")
    print(df_rep.head(top_k)[["feature", "value_transformed", "shap"]].to_string(index=False))
    print()

    pos_df = df_rep[df_rep["shap"] > 0].head(top_k)
    neg_df = df_rep[df_rep["shap"] < 0].head(top_k)

    print(f"Top positive drivers (up):")
    if len(pos_df):
        print(pos_df[["feature", "value_transformed", "shap"]].to_string(index=False))
    else:
        print("(none)")
    print()

    print(f"Top negative drivers (down):")
    if len(neg_df):
        print(neg_df[["feature", "value_transformed", "shap"]].to_string(index=False))
    else:
        print("(none)")
    print("=" * 80)

def fit_multihot_rate_encoding(X: pd.DataFrame, y: np.ndarray, cols: list[str], smooth: float = 20.0):
    """
    Fit smoothed per-label target rates for a multilabel indicator block.

    Args:
        X: Feature dataframe containing 0/1 indicator columns.
        y: Binary target array.
        cols: Indicator columns to encode.
        smooth: Smoothing strength toward the global target mean.

    Returns:
        A tuple of:
            - smoothed per-column mean rates
            - per-column counts
            - global target mean
    """
    M = X[cols].to_numpy(dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    counts = M.sum(axis=0)
    sums = (M * y[:, None]).sum(axis=0)

    global_mean = float(y.mean()) if len(y) else 0.0
    means = (sums + smooth * global_mean) / (counts + smooth)
    return means.astype(np.float32), counts.astype(np.float32), global_mean


def apply_multihot_rate_features(
    X: pd.DataFrame,
    cols: list[str],
    means: np.ndarray,
    counts: np.ndarray,
    global_mean: float,
    prefix: str,
) -> pd.DataFrame:
    """
    Add dense summary features from a fitted multilabel rate encoding.

    Args:
        X: Feature dataframe to augment.
        cols: Source 0/1 indicator columns.
        means: Smoothed per-column target rates.
        counts: Per-column supports.
        global_mean: Global target mean from the training set.
        prefix: Prefix for output feature names.

    Returns:
        The input dataframe with `{prefix}_mean_rate`, `{prefix}_mean_support`,
        and `{prefix}_present` appended.
    """
    M = X[cols].to_numpy(dtype=np.float32)
    present = M.sum(axis=1)
    denom = np.maximum(present, 1.0)

    mean_rate = (M @ means) / denom
    mean_support = (M @ counts) / denom

    X[f"{prefix}_mean_rate"] = mean_rate
    X[f"{prefix}_mean_support"] = mean_support
    X[f"{prefix}_present"] = present
    return X

def fit_multihot_resid_encoding(X: pd.DataFrame, y: np.ndarray, cols: list[str], smooth: float = 10.0):
    """
    Fit smoothed per-label residual means for a multilabel indicator block.

    Args:
        X: Feature dataframe containing 0/1 indicator columns.
        y: Residual regression target.
        cols: Indicator columns to encode.
        smooth: Smoothing strength toward the global residual mean.

    Returns:
        A tuple of:
            - smoothed per-column residual means
            - per-column counts
            - global residual mean
    """
    M = X[cols].to_numpy(dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    counts = M.sum(axis=0)
    sums = (M * y[:, None]).sum(axis=0)
    global_mean = float(y.mean()) if len(y) else 0.0

    means = (sums + smooth * global_mean) / (counts + smooth)

    return means.astype(np.float32), counts.astype(np.float32), global_mean


def apply_multihot_resid_features(
    X: pd.DataFrame,
    cols: list[str],
    means: np.ndarray,
    counts: np.ndarray,
    global_mean: float,
    prefix: str,
) -> pd.DataFrame:
    """
    Add dense summary features from a fitted multilabel residual encoding.

    Args:
        X: Feature dataframe to augment.
        cols: Source 0/1 indicator columns.
        means: Smoothed per-column residual means.
        counts: Per-column supports.
        global_mean: Global residual mean from the training set.
        prefix: Prefix for output feature names.

    Returns:
        The input dataframe with `{prefix}_mean_resid`, `{prefix}_mean_support`,
        and `{prefix}_present` appended.
    """
    M = X[cols].to_numpy(dtype=np.float32)
    present = M.sum(axis=1)

    denom = np.maximum(present, 1.0)

    mean_resid = (M @ means) / denom
    mean_support = (M @ counts) / denom

    X[f"{prefix}_mean_resid"] = mean_resid
    X[f"{prefix}_mean_support"] = mean_support
    X[f"{prefix}_present"] = present

    return X

def coerce_boolish_to_str(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool).map({True: "TRUE", False: "FALSE"})


def build_features(df_in: pd.DataFrame, include_list_years: bool, use_stats: bool, stats_mode: str = "finish"):
    """
    Build the base tabular feature set for the score or finish model.

    Args:
        df_in: Input dataframe containing AniList metadata and optional
            personal-list-derived columns.
        include_list_years: Whether to include user list year fields such as
            start and completion year.
        use_stats: Whether to parse AniList community-distribution fields.
        stats_mode: Feature mode, typically `score` or `finish`.

    Returns:
        A tuple of:
            - dataframe of engineered base features
            - list of categorical columns intended for one-hot encoding

    Notes:
        - Genre and tag indicators are added outside this function.
        - Missing numeric values are coerced to 0.
        - Missing categorical values are filled with `UNKNOWN`.
        - Community-derived score and finish features are controlled here.
    """
    df = df_in.copy()

    df["is_finished"] = (
        (pd.to_numeric(df.get("endDate_year", 0), errors="coerce").fillna(0) > 0) |
        (df.get("status", "").astype(str).str.upper() == "FINISHED")
    ).astype(int)

    core_num_cols = [
        "episodes", "duration", "startDate_year", "endDate_year",
        "popularity", "favourites", "trending",
        "meanScore",
    ]
    list_year_cols = ["StartYear_list", "CompletedYear_list"]

    for c in core_num_cols + (list_year_cols if include_list_years else []):
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["total_minutes"] = (df["episodes"] * df["duration"]).clip(lower=0)
    df["log1p_total_minutes"] = np.log1p(df["total_minutes"])


    for c in ["popularity", "favourites", "trending"]:
        df[f"log1p_{c}"] = np.log1p(df[c].clip(lower=0))

    if USE_COMMUNITY_CONTEXT:
        df["meanScore_10"] = df["meanScore"] / 10.0
        df["mean_x_pop"] = df["meanScore_10"] * df["log1p_popularity"]
    else:
        df["meanScore_10"] = 0.0
        df["mean_x_pop"] = 0.0

    if "isAdult" not in df.columns:
        df["isAdult"] = False
    if "isLicensed" not in df.columns:
        df["isLicensed"] = False

    df["isAdult_str"] = coerce_boolish_to_str(df["isAdult"])
    df["isLicensed_str"] = coerce_boolish_to_str(df["isLicensed"])
    df["startDate_year"] = pd.to_numeric(df.get("startDate_year", 0), errors="coerce").fillna(0)
    df["endDate_year"]   = pd.to_numeric(df.get("endDate_year", 0), errors="coerce").fillna(0)
    df["startYear_bin"] = year_bin(df["startDate_year"])
    df["endYear_bin"]   = year_bin(df["endDate_year"])
    cat_cols = [
        "format", "source", "countryOfOrigin",
        "isAdult_str", "isLicensed_str", "startYear_bin", 
    ]
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "UNKNOWN"
        df[c] = df[c].fillna("UNKNOWN").astype(str)

    base_num = [
        "log1p_total_minutes",
        "log1p_popularity", 
        "meanScore_10",
    ]


    if include_list_years:
        for c in list_year_cols:
            if c not in df.columns:
                df[c] = 0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        base_num += list_year_cols

    if use_stats:
        stats_df = parse_status_score_stats(df)
        df = pd.concat([df, stats_df], axis=1)

        engagement_cols = ["stats_p_completed", "stats_p_dropped", "stats_retention", "stats_hype_gap"]
        for c in engagement_cols:
            if c in df.columns:
                df.loc[df["is_finished"] == 0, c] = 0.0

        if stats_mode == "score":
            base_num += [
                
                
            ]
            base_num += ["is_finished"]

        else:
            base_num += [
                "stats_log1p_n_total",
                "stats_p_completed",
                "stats_p_dropped",
                "stats_retention",
                "stats_hype_gap",
                
                
                "stats_p90plus",
                "stats_p50minus",
                
            ]

    X_base = df[base_num + cat_cols].copy()
    return X_base, cat_cols

MIN_FREQ_STUDIO = 5

def parse_studios_series(series: pd.Series) -> list[list[str]]:
    """
    Parse AniList studio payloads into studio-name lists.

    Args:
        series: Pandas Series containing JSON arrays of studio objects or strings.

    Returns:
        A list where each element is a list of studio names for that row.
    """
    out = []
    for v in series:
        arr = safe_json_loads(v, default=[])
        if isinstance(arr, list):
            names = []
            for item in arr:
                if isinstance(item, dict) and "name" in item:
                    names.append(str(item["name"]))
                elif isinstance(item, str):
                    names.append(item)
            out.append(names)
        else:
            out.append([])
    return out

class _PrepXGB:
    """
    Lightweight wrapper that applies a fitted preprocessor before an XGBoost classifier.

    This makes the combined object behave like a sklearn-style classifier with
    `predict_proba` and `predict` methods operating on raw dataframes.
    """
    _estimator_type = "classifier" 

    def __init__(self, prep, model):
        self.prep = prep
        self.model = model
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None, **kwargs):
        return self

    def predict_proba(self, X):
        Xt = self.prep.transform(X)
        try:
            if not sparse.issparse(Xt):
                Xt = sparse.csr_matrix(Xt)
        except Exception:
            pass

        if hasattr(self.model, "best_iteration") and self.model.best_iteration is not None:
            try:
                p1 = self.model.predict_proba(
                    Xt, iteration_range=(0, int(self.model.best_iteration) + 1)
                )[:, 1]
            except TypeError:
                p1 = self.model.predict_proba(Xt)[:, 1]
        else:
            p1 = self.model.predict_proba(Xt)[:, 1]

        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)


def fit_finish_xgb_with_early_stopping(
    X_train_df, y_train, w_train,
    prep_finish,
    random_state=42,
    test_size=0.15,
    early_rounds=200,
):
    """
    Fit the finish classifier with an internal validation split and early stopping.

    Args:
        X_train_df: Training feature dataframe.
        y_train: Binary drop target where 1 means drop-risk and 0 means finish.
        w_train: Sample weights for class imbalance and inferred-label weighting.
        prep_finish: Unfitted preprocessing pipeline for finish features.
        random_state: Random seed for splitting and model fitting.
        test_size: Fraction of input rows reserved for validation.
        early_rounds: Number of non-improving rounds before stopping.

    Returns:
        A `_PrepXGB` wrapper containing the fitted preprocessor and classifier.
    """
    strat = y_train if len(np.unique(y_train)) > 1 else None
    X_tr_df, X_va_df, y_tr, y_va, w_tr, w_va = train_test_split(
        X_train_df, y_train, w_train,
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )

    prep = clone(prep_finish)
    prep.fit(X_tr_df)

    X_tr = prep.transform(X_tr_df)
    X_va = prep.transform(X_va_df)

    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",

        n_estimators=50000,
        learning_rate=0.015,

        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,

        reg_lambda=10.0,
        reg_alpha=0.5,
        min_child_weight=10,

        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,

        callbacks=[EarlyStopping(rounds=early_rounds, save_best=True)],
    )

    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    return _PrepXGB(prep, model)

def confidence_label(u: float) -> str:
    """
    Convert a normalized uncertainty score into a coarse confidence label.

    Args:
        u: Uncertainty value on an approximate 0 to 1 scale.

    Returns:
        `High`, `Medium`, or `Experimental`.
    """
    if u < 0.33:
        return "High"
    if u < 0.66:
        return "Medium"
    return "Experimental"

def time_based_split_indices(dates: pd.Series, test_fraction: float = 0.20):
    """
    Split row indices into chronological train and test partitions.

    Args:
        dates: Date-like series used for chronological ordering.
        test_fraction: Fraction of the most recent rows assigned to test.

    Returns:
        A tuple of `(train_idx, test_idx)` as NumPy index arrays.

    Notes:
        Missing dates are filled with an old sentinel date so they sort first.
    """
    d = pd.to_datetime(dates, errors="coerce")
    d_filled = d.fillna(pd.Timestamp("1970-01-01"))

    order = np.argsort(d_filled.values.astype("datetime64[ns]"))
    n = len(order)
    n_test = max(1, int(np.floor(n * test_fraction)))
    test_idx = order[-n_test:]
    train_idx = order[:-n_test]

    return train_idx, test_idx

def main():
    """
    Run the full personalized AniList training and prediction pipeline.

    The pipeline:
    - loads the personal-list export and AniList dataset
    - builds score and finish training sets
    - engineers metadata, tag, genre, and community features
    - trains and evaluates the score model
    - trains and calibrates the finish model
    - generates explanations, confidence labels, and final rankings
    - writes the final prediction CSV
    """
    print("Loading datasets...")
    personal = pd.read_csv(PERSONAL_CSV)
    anilist = pd.read_csv(ANILIST_CSV)

    if "AniList ID" not in personal.columns:
        raise ValueError("Personal CSV must contain column 'AniList ID'.")
    if "Status" not in personal.columns:
        raise ValueError("Personal CSV must contain column 'Status'.")

    personal["AniList ID"] = pd.to_numeric(personal["AniList ID"], errors="coerce")
    personal = personal.dropna(subset=["AniList ID"]).copy()
    personal["AniList ID"] = personal["AniList ID"].astype(int)

    anilist["id"] = pd.to_numeric(anilist["id"], errors="coerce")
    anilist = anilist.dropna(subset=["id"]).copy()
    anilist["id"] = anilist["id"].astype(int)

    personal["Status"] = personal["Status"].fillna("UNKNOWN").astype(str).str.upper()
    personal["Progress_int"] = parse_progress_col(personal.get("Progress", pd.Series([0] * len(personal))))
    personal["StartDT"] = safe_start_date(personal.get("Start Date", pd.Series([pd.NA] * len(personal))))
    personal["CompletedDT"] = safe_start_date(personal.get("Completed Date", pd.Series([pd.NA] * len(personal))))

    if "Your_Score" in personal.columns:
        personal["Your_Score"] = pd.to_numeric(personal["Your_Score"], errors="coerce")
    else:
        personal["Your_Score"] = pd.to_numeric(personal.get("Score", np.nan), errors="coerce")

    personal.loc[personal["Your_Score"].fillna(0) <= 0, "Your_Score"] = np.nan

    personal["Score10"] = (personal["Your_Score"] / 10.0).clip(1, 10)

    scored = personal[personal["Score10"].notna()].copy()
    if len(scored) == 0:
        raise ValueError("No scored entries found (Your_Score > 0).")
    personal["StartYear_list"] = parse_date_to_year(personal.get("Start Date", pd.Series([pd.NA] * len(personal))))
    personal["CompletedYear_list"] = parse_date_to_year(personal.get("Completed Date", pd.Series([pd.NA] * len(personal))))

    rated_all = personal.merge(anilist, left_on="AniList ID", right_on="id", how="inner", suffixes=("_list", "_ani")).copy()
    rated_scored = scored.merge(anilist, left_on="AniList ID", right_on="id", how="inner", suffixes=("_list", "_ani")).copy()
    n_missing_mean = int((pd.to_numeric(rated_scored["meanScore"], errors="coerce").fillna(0) <= 0).sum())
    print("Scored rows with meanScore<=0:", n_missing_mean)

    rated_scored = rated_scored.merge(
        personal[["AniList ID", "StartYear_list", "CompletedYear_list"]],
        on="AniList ID",
        how="left"
    )
    rated_scored["StartYear_list"] = rated_scored["StartYear_list"].fillna(0).astype(int)
    rated_scored["CompletedYear_list"] = rated_scored["CompletedYear_list"].fillna(0).astype(int)

    mask_valid = is_valid_longform_anime(rated_all)
    rated_all = rated_all[mask_valid].copy()

    mask_valid_scored = is_valid_longform_anime(rated_scored)
    rated_scored = rated_scored[mask_valid_scored].copy()

    print(f"Matched list entries: {len(rated_all)}")
    print(f"Matched scored entries: {len(rated_scored)}")

    if ONLY_COMPLETED_FOR_SCORE_TRAIN:
        rated_scored = rated_scored[rated_scored["Status"].astype(str).str.upper() == "COMPLETED"].copy()
        print(f"Score training rows after ONLY_COMPLETED filter: {len(rated_scored)}")

    for df in (rated_all, rated_scored, anilist):
        for col, default in [
            ("genres", "[]"),
            ("tags", "[]"),
            ("format", "UNKNOWN"),
            ("source", "UNKNOWN"),
            ("status", "UNKNOWN"),
            ("season", "UNKNOWN"),
            ("countryOfOrigin", "UNKNOWN"),
            ("isAdult", False),
            ("isLicensed", False),
            ("episodes", 0),
            ("duration", 0),
            ("startDate_year", 0),
            ("endDate_year", 0),
            ("popularity", 0),
            ("favourites", 0),
            ("trending", 0),
            ("meanScore", 0),
            ("averageScore", 0),
        ]:
            if col not in df.columns:
                df[col] = default

    anilist_valid = anilist[is_valid_longform_anime(anilist)].copy()
    anilist_valid = anilist_valid.reset_index(drop=True)
    anilist_pos = pd.Series(
        anilist_valid.index.values,
        index=anilist_valid["id"].values
    )


    rated_all["Status"] = rated_all["Status"].fillna("UNKNOWN").astype(str).str.upper()

    today = pd.Timestamp.today().normalize()

    completed_for_pace = rated_all[rated_all["Status"].astype(str).str.upper() == "COMPLETED"].copy()
    dpe_by_bucket = build_days_per_episode_by_bucket(completed_for_pace)
    print("Learned pace (median days/episode) by bucket:", dpe_by_bucket)

    rated_all["finish_label"] = rated_all.apply(
        lambda r: infer_finish_label(
            r,
            today=today,
            dpe_by_bucket=dpe_by_bucket,
            min_days_floor=FINISH_MIN_DAYS_FLOOR,
            multiplier=FINISH_MULTIPLIER,
            min_progress_fraction=FINISH_MIN_PROGRESS_FRACTION
        ),
        axis=1
    )

    finish_train = rated_all[rated_all["finish_label"].notna()].copy()
    finish_train["finish_label"] = finish_train["finish_label"].astype(int)

    finish_train["is_true_dropped"] = (finish_train["Status"] == "DROPPED").astype(int)
    finish_train["is_inferred_drop"] = (
        (finish_train["finish_label"] == 0) & (finish_train["Status"].isin(["CURRENT", "PAUSED", "ON_HOLD"]))
    ).astype(int)
    n_true_dropped = int((rated_all["Status"] == "DROPPED").sum())
    n_inferred = int(
        (rated_all["Status"].isin(["CURRENT", "PAUSED", "ON_HOLD"]) &
        (rated_all["finish_label"] == 0)).sum()
    )

    print(
        f"Finish training rows (labeled): {len(finish_train)} "
        f"(completed={int((finish_train['finish_label']==1).sum())}, "
        f"dropped/inferred={int((finish_train['finish_label']==0).sum())})"
    )
    print(f"Inferred drops added: {n_inferred} (true DROPPED in list: {n_true_dropped})")
    print("Finish label distribution:")
    print(finish_train["finish_label"].value_counts())

    if finish_train["finish_label"].nunique() < 2:
        raise ValueError(
            "Finish training labels have only one class. "
            "You may need more DROPPED entries or relax inference thresholds."
        )


    print("Parsing genres and tags...")
    scored_genres = parse_genres_series(rated_scored["genres"])
    scored_tags   = parse_tags_series(rated_scored["tags"])

    finish_genres = parse_genres_series(finish_train["genres"])
    finish_tags   = parse_tags_series(finish_train["tags"])

    all_genres = parse_genres_series(anilist_valid["genres"])
    all_tags   = parse_tags_series(anilist_valid["tags"])

    print("Encoding genres/tags...")

    scored_genre_df = multilabel_to_indicator_frame(
        scored_genres, prefix="genre_", min_freq=MIN_FREQ_GENRE
    )
    scored_tag_df = multilabel_to_indicator_frame(
        scored_tags, prefix="tag_", min_freq=MIN_FREQ_TAG_SCORE
    )

    all_genre_df_score = align_indicator_columns(
        all_genres, scored_genre_df.columns.tolist(), prefix="genre_"
    )
    all_tag_df_score = align_indicator_columns(
        all_tags, scored_tag_df.columns.tolist(), prefix="tag_"
    )

    MIN_FREQ_TAG_FINISH = 10
    MIN_FREQ_GENRE_FINISH = MIN_FREQ_GENRE

    finish_genre_vocab = multilabel_to_indicator_frame(
        finish_genres, prefix="genre_", min_freq=MIN_FREQ_GENRE_FINISH
    )
    finish_tag_vocab = multilabel_to_indicator_frame(
        finish_tags, prefix="tag_", min_freq=MIN_FREQ_TAG_FINISH
    )

    finish_genre_df = finish_genre_vocab
    finish_tag_df   = finish_tag_vocab

    all_genre_df_finish = align_indicator_columns(
        all_genres, finish_genre_vocab.columns.tolist(), prefix="genre_"
    )
    all_tag_df_finish = align_indicator_columns(
        all_tags, finish_tag_vocab.columns.tolist(), prefix="tag_"
    )



    X_score_base, cat_cols_score = build_features(
        rated_scored,
        include_list_years=False,
        use_stats=True,
        stats_mode="score",
    )
    y_score = (rated_scored["Score10"] - rated_scored["meanScore"]/10.0).values

    X_finish_base, cat_cols_finish = build_features(
        finish_train,
        include_list_years=False,
        use_stats=True,
        stats_mode="finish",
    )
    y_drop = (finish_train["finish_label"].to_numpy(dtype=int) == 0).astype(int)

    w_drop = np.ones(len(finish_train), dtype=np.float32)

    INFERRED_DROP_WEIGHT = 0.6
    w_drop[finish_train["is_inferred_drop"].to_numpy(dtype=bool)] = INFERRED_DROP_WEIGHT
    TRUE_DROP_WEIGHT = 1.2
    w_drop[finish_train["is_true_dropped"].to_numpy(dtype=bool)] = TRUE_DROP_WEIGHT


    X_all_base_score, _ = build_features(
        anilist_valid,
        include_list_years=False,
        use_stats=True,
        stats_mode="score",
    )
    X_all_base_finish, _ = build_features(
        anilist_valid,
        include_list_years=False,
        use_stats=True,
        stats_mode="finish",
    )

    X_score     = pd.concat([X_score_base.reset_index(drop=True), scored_genre_df, scored_tag_df], axis=1)
    X_all_score = pd.concat([X_all_base_score.reset_index(drop=True), all_genre_df_score, all_tag_df_score], axis=1)

    X_finish     = pd.concat([X_finish_base.reset_index(drop=True), finish_genre_df, finish_tag_df], axis=1)
    X_all_finish = pd.concat([X_all_base_finish.reset_index(drop=True), all_genre_df_finish, all_tag_df_finish], axis=1)


    X_score = X_score.loc[:, ~X_score.columns.duplicated()].copy()
    X_finish = X_finish.loc[:, ~X_finish.columns.duplicated()].copy()
    X_all_score = X_all_score.loc[:, ~X_all_score.columns.duplicated()].copy()
    X_all_finish = X_all_finish.loc[:, ~X_all_finish.columns.duplicated()].copy()

    X_all_finish = X_all_finish.reindex(columns=X_finish.columns, fill_value=0)
    X_all_score = X_all_score.reindex(columns=X_score.columns, fill_value=0)
    raw_finish_onehots = [c for c in X_finish.columns if c.startswith(("tag_", "genre_"))]
    raw_finish_onehots = [c for c in raw_finish_onehots if not c.startswith(("tag_drop_", "genre_drop_"))]

    X_finish = X_finish.drop(columns=raw_finish_onehots, errors="ignore")
    X_all_finish = X_all_finish.drop(columns=raw_finish_onehots, errors="ignore")

    def split_dense_vs_binary(X: pd.DataFrame, cat_cols: list[str], dense_candidates: list[str]):
        dense = [c for c in dense_candidates if c in X.columns]
        binary = [c for c in X.columns if c not in cat_cols and c not in dense]
        return dense, binary

    dense_candidates_score = [
        "episodes", "duration",
        "log1p_total_minutes",
        "log1p_popularity",  
        "meanScore_10",
         "tag_mean_resid", "genre_mean_resid",
    ]


    dense_candidates_finish = [
        "episodes", "duration", "endDate_year",
        "log1p_popularity",  
        "meanScore_10",
        "StartYear_list", "CompletedYear_list",
        "stats_log1p_n_total","stats_p_completed","stats_p_dropped","stats_retention",
        "stats_hype_gap","stats_p90plus","stats_p50minus",
         "log1p_total_minutes"
    ]

    dense_num_score, binary_num_score = split_dense_vs_binary(X_score, cat_cols_score, dense_candidates_score)
    dense_num_finish, binary_num_finish = split_dense_vs_binary(X_finish, cat_cols_finish, dense_candidates_finish)

    print(f"[DEBUG] score dense={len(dense_num_score)} binary={len(binary_num_score)}")
    print(f"[DEBUG] finish dense={len(dense_num_finish)} binary={len(binary_num_finish)}")

    prep_score = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_score),
        ("dense", "passthrough", dense_num_score),
        ("bin", "passthrough", binary_num_score),
    ], remainder="drop")

    dense_candidates_finish = [
        "episodes", "duration", "endDate_year",
        "log1p_popularity",  
        "meanScore_10",
        "stats_log1p_n_total","stats_p_completed","stats_p_dropped","stats_retention",
        "stats_hype_gap","stats_p90plus","stats_p50minus",
         "log1p_total_minutes"
    ]

    dense_num_finish, binary_num_finish = split_dense_vs_binary(X_finish, cat_cols_finish, dense_candidates_finish)

    prep_finish = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_finish),
        ("dense", "passthrough", dense_num_finish),
        ("bin", "passthrough", binary_num_finish),
    ], remainder="drop")






    print("\nTraining score model (XGBoost)...")

    tag_cols   = [c for c in X_score.columns if c.startswith("tag_")]
    genre_cols = [c for c in X_score.columns if c.startswith("genre_")]

    if SCORE_SPLIT_MODE == "time":
        score_dates = pd.to_datetime(rated_scored["CompletedDT"], errors="coerce")
        score_dates = score_dates.fillna(pd.to_datetime(rated_scored["StartDT"], errors="coerce"))
        tr_i, te_i = time_based_split_indices(score_dates.reset_index(drop=True), test_fraction=TEST_FRACTION)

        Xs_tr = X_score.iloc[tr_i].copy()
        Xs_te = X_score.iloc[te_i].copy()
        ys_tr = y_score[tr_i]
        ys_te = y_score[te_i]
    else:
        Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
            X_score, y_score, test_size=TEST_FRACTION, random_state=42
        )

    tag_means, tag_counts, tag_global = fit_multihot_resid_encoding(Xs_tr, ys_tr, tag_cols, smooth=30.0)
    gen_means, gen_counts, gen_global = fit_multihot_resid_encoding(Xs_tr, ys_tr, genre_cols, smooth=1.0)

    Xs_tr = apply_multihot_resid_features(Xs_tr.copy(), tag_cols,   tag_means, tag_counts, tag_global, "tag")
    Xs_te = apply_multihot_resid_features(Xs_te.copy(), tag_cols,   tag_means, tag_counts, tag_global, "tag")
    Xs_tr = apply_multihot_resid_features(Xs_tr,        genre_cols, gen_means, gen_counts, gen_global, "genre")
    Xs_te = apply_multihot_resid_features(Xs_te,        genre_cols, gen_means, gen_counts, gen_global, "genre")

    Xs_te = Xs_te.reindex(columns=Xs_tr.columns, fill_value=0)

    print(f"Score split ({SCORE_SPLIT_MODE}): train={len(Xs_tr)} test={len(Xs_te)}")

    tr_idx = Xs_tr.index.to_numpy()
    te_idx = Xs_te.index.to_numpy()

    dense_candidates_score = [
        "episodes", "duration",
        "log1p_popularity",  
        "meanScore_10", 
        "tag_mean_resid", "genre_mean_resid",
    ]

    dense_num_score, binary_num_score = split_dense_vs_binary(Xs_tr, cat_cols_score, dense_candidates_score)

    prep_score = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_score),
        ("dense", "passthrough", dense_num_score),
        ("bin", "passthrough", binary_num_score),
    ], remainder="drop")

    prep_score.fit(Xs_tr)
    Xs_tr_t = prep_score.transform(Xs_tr)
    Xs_te_t = prep_score.transform(Xs_te)

    xgb_score = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        max_bin=512,

        n_estimators=50000,
        learning_rate=0.02,

        grow_policy="lossguide",
        max_leaves=96,
        max_depth=0,
        min_child_weight=5,

        subsample=0.85,
        colsample_bytree=0.75,
        colsample_bynode=0.90,

        reg_lambda=5.0,
        reg_alpha=0.05,
        gamma=0.00,

        max_delta_step=1.0,
        random_state=42,
        n_jobs=-1,

        eval_metric="rmse",
        callbacks=[EarlyStopping(rounds=200, save_best=True)],
    )



    xgb_score.fit(
        Xs_tr_t,
        ys_tr,
        eval_set=[(Xs_te_t, ys_te)],
        verbose=False,
    )


    train_resid_pred = xgb_score.predict(Xs_tr_t)
    resid_bias = float(train_resid_pred.mean() - ys_tr.mean())
    print("Residual bias (0–10):", resid_bias)
    class _XGBWrapped:
        def __init__(self, prep, model):
            self.prep = prep
            self.model = model
        def fit(self, X, y):
            return self
        def predict(self, X):
            Xt = self.prep.transform(X)
            return self.model.predict(Xt)

    wrapped_score = _XGBWrapped(prep_score, xgb_score)

    export_permutation_importance(
        wrapped_score,
        X=Xs_te,
        y=ys_te,
        out_csv="feature_importance_score_perm.csv",
        top_k=60,
        title="Score model (XGBoost)",
        n_repeats=8,
        max_rows=2000,
        scoring="neg_root_mean_squared_error",
    )


    pred_resid_raw_te = xgb_score.predict(Xs_te_t).astype(np.float32)
    pred_resid_te = pred_resid_raw_te - np.float32(resid_bias)

    err_resid = pred_resid_te - ys_te.astype(np.float32)

    rmse = float(np.sqrt(np.mean(err_resid ** 2)))
    mae  = float(np.mean(np.abs(err_resid)))
    p95  = float(np.percentile(np.abs(err_resid), 95))
    p99  = float(np.percentile(np.abs(err_resid), 99))
    mx   = float(np.max(np.abs(err_resid)))

    print(f"[TEST residual 0–10] RMSE={rmse:.3f}  MAE={mae:.3f}  p95={p95:.3f}  p99={p99:.3f}  max={mx:.3f}")

    mean_score_10_scored = (
        pd.to_numeric(rated_scored["meanScore"], errors="coerce")
        .fillna(0).to_numpy(dtype=np.float32) / 10.0
    )
    true10_scored = (
        pd.to_numeric(rated_scored["Score10"], errors="coerce")
        .fillna(0).to_numpy(dtype=np.float32)
    )

    crowd10_te = mean_score_10_scored[te_idx]
    true100_te = true10_scored[te_idx] * 10.0

    pred10_te = np.clip(crowd10_te + pred_resid_te, 1.0, 10.0)
    pred100_te = pred10_te * 10.0

    err100 = pred100_te - true100_te
    abs_err100 = np.abs(err100)

    print(f"[TEST score 0–100] bias={err100.mean():+.2f}  MAE={abs_err100.mean():.2f}  "
        f"p95={np.percentile(abs_err100,95):.2f}  p99={np.percentile(abs_err100,99):.2f}  "
        f"max={abs_err100.max():.2f}")
    try:
        pred100_te_cal = calibrate_score100(pred100_te)
        mae_te_cal = float(np.mean(np.abs(pred100_te_cal - true100_te)))
        print(f"[TEST score 0–100] MAE calibrated={mae_te_cal:.2f}")
    except Exception:
        pass

    sp = spearmanr(pred100_te, true100_te).correlation
    pe = pearsonr(pred100_te, true100_te)[0]
    print(f"[TEST score 0–100] Spearman={sp:.3f}  Pearson={pe:.3f}")

    ids_scored = pd.to_numeric(rated_scored["id"], errors="coerce").fillna(0).astype(int).to_numpy()
    score_train_ids = set(ids_scored[tr_idx])
    score_test_ids  = set(ids_scored[te_idx])

    best_n = int(getattr(xgb_score, "best_iteration", xgb_score.n_estimators - 1) + 1)
    print("Score best_n_estimators:", best_n)

    tag_means_full, tag_counts_full, tag_global_full = fit_multihot_resid_encoding(
        X_score, y_score, tag_cols, smooth=10.0
    )
    gen_means_full, gen_counts_full, gen_global_full = fit_multihot_resid_encoding(
        X_score, y_score, genre_cols, smooth=5.0
    )

    X_score_full = X_score.copy()
    X_score_full = apply_multihot_resid_features(
        X_score_full, tag_cols, tag_means_full, tag_counts_full, tag_global_full, "tag"
    )
    X_score_full = apply_multihot_resid_features(
        X_score_full, genre_cols, gen_means_full, gen_counts_full, gen_global_full, "genre"
    )

    X_all_score = X_all_score.copy()
    X_all_score = apply_multihot_resid_features(
        X_all_score, tag_cols, tag_means_full, tag_counts_full, tag_global_full, "tag"
    )
    X_all_score = apply_multihot_resid_features(
        X_all_score, genre_cols, gen_means_full, gen_counts_full, gen_global_full, "genre"
    )

    X_all_score = X_all_score.reindex(columns=X_score_full.columns, fill_value=0)

    dense_candidates_score_final = [
        "episodes", "duration",
        "log1p_popularity",  
        "meanScore_10", 
        "tag_mean_resid", "genre_mean_resid",
    ]
    dense_num_score, binary_num_score = split_dense_vs_binary(
        X_score_full, cat_cols_score, dense_candidates_score_final
    )

    dense_num_score = dense_num_score
    tag_means_all, tag_counts_all, tag_global_all = fit_multihot_resid_encoding(
        X_score, y_score, tag_cols, smooth=10.0
    )
    gen_means_all, gen_counts_all, gen_global_all = fit_multihot_resid_encoding(
        X_score, y_score, genre_cols, smooth=5.0
    )

    X_score_full = apply_multihot_resid_features(X_score.copy(), tag_cols, tag_means_all, tag_counts_all, tag_global_all, "tag")
    X_score_full = apply_multihot_resid_features(X_score_full, genre_cols, gen_means_all, gen_counts_all, gen_global_all, "genre")

    X_all_score_full = apply_multihot_resid_features(X_all_score.copy(), tag_cols, tag_means_all, tag_counts_all, tag_global_all, "tag")
    X_all_score_full = apply_multihot_resid_features(X_all_score_full, genre_cols, gen_means_all, gen_counts_all, gen_global_all, "genre")

    X_all_score_full = X_all_score_full.reindex(columns=X_score_full.columns, fill_value=0)


    prep_score_final = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_score),
        ("dense", "passthrough", dense_num_score),
        ("bin", "passthrough", binary_num_score),
    ], remainder="drop")

    prep_score_final.fit(X_score_full)
    X_score_full_t = prep_score_final.transform(X_score_full)

    final_params = xgb_score.get_params()
    final_params["n_estimators"] = best_n
    final_params["callbacks"] = None

    xgb_score_final = XGBRegressor(**final_params)
    xgb_score_final.fit(X_score_full_t, y_score, verbose=False)
    pred_resid_train = xgb_score_final.predict(X_score_full_t).astype(np.float32) - np.float32(resid_bias)
    crowd10_train = (pd.to_numeric(rated_scored["meanScore"], errors="coerce").fillna(0).to_numpy(np.float32) / 10.0)
    true10_train  = pd.to_numeric(rated_scored["Score10"], errors="coerce").fillna(0).to_numpy(np.float32)
    pred10_train  = np.clip(crowd10_train + pred_resid_train, 1.0, 10.0)

    mae_train_100 = float(np.mean(np.abs(pred10_train * 10.0 - true10_train * 10.0)))
    print(f"[TRAIN score 0–100] MAE={mae_train_100:.2f}")

    pred_all = xgb_score_final.predict(X_score_full_t).astype(np.float32)
    resid_bias = float(pred_all.mean() - np.asarray(y_score, dtype=np.float32).mean())
    print("Residual bias after final refit (0–10):", resid_bias)

    prep_score = prep_score_final
    xgb_score = xgb_score_final
    crowd10_train = (
        pd.to_numeric(rated_scored["meanScore"], errors="coerce")
        .fillna(0).to_numpy(dtype=np.float32) / 10.0
    )
    true100_train = (
        pd.to_numeric(rated_scored["Score10"], errors="coerce")
        .fillna(0).to_numpy(dtype=np.float32) * 10.0
    )

    pred_resid_train = xgb_score.predict(X_score_full_t).astype(np.float32) - np.float32(resid_bias)
    pred10_train = np.clip(crowd10_train + pred_resid_train, 1.0, 10.0)
    pred100_train = pred10_train * 10.0

    mx = float(pred100_train.mean())
    my = float(true100_train.mean())
    cov = float(np.mean((pred100_train - mx) * (true100_train - my)))
    var = float(np.mean((pred100_train - mx) ** 2)) + 1e-6

    score_cal_b = cov / var
    score_cal_b = float(np.clip(score_cal_b, 0.85, 1.25))

    score_cal_a = float(my - score_cal_b * mx)

    def calibrate_score100(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return np.clip(score_cal_a + score_cal_b * x, 1.0, 100.0)

    mae_train_raw = float(np.mean(np.abs(pred100_train - true100_train)))
    mae_train_cal = float(np.mean(np.abs(calibrate_score100(pred100_train) - true100_train)))

    print(f"[SCORE CAL] a={score_cal_a:+.3f}  b={score_cal_b:.3f}")
    print(f"[TRAIN score 0–100] MAE raw={mae_train_raw:.2f}  cal={mae_train_cal:.2f}")

    print("\nPrecomputing predicted user scores for finish model features...")

    mean_score_10_all = (
        pd.to_numeric(anilist_valid["meanScore"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32) / 10.0
    )

    pred_resid_all = xgb_score.predict(prep_score.transform(X_all_score)).astype(np.float32) - np.float32(resid_bias)

    pred_user_score10_all_raw = np.clip(mean_score_10_all + pred_resid_all, 1.0, 10.0)
    pred_user_score100_all = calibrate_score100(pred_user_score10_all_raw * 10.0)
    pred_user_score10_all = pred_user_score100_all / 10.0


    id_to_pred_user_score10 = pd.Series(
        pred_user_score10_all, index=anilist_valid["id"].astype(int).to_numpy()
    )

    finish_ids = finish_train["id"].reset_index(drop=True).astype(int)
    X_finish["pred_user_score10"] = finish_ids.map(id_to_pred_user_score10).fillna(0.0).to_numpy(dtype=np.float32)

    X_all_finish["pred_user_score10"] = pred_user_score10_all.astype(np.float32)

    print("\nTraining finish model...")

    if FINISH_SPLIT_MODE == "time":
        finish_dates = pd.to_datetime(finish_train["CompletedDT"], errors="coerce")
        finish_dates = finish_dates.fillna(pd.to_datetime(finish_train["StartDT"], errors="coerce"))
        finish_dates = finish_dates.reset_index(drop=True)

        tr_i, te_i = time_based_split_indices(finish_dates, test_fraction=TEST_FRACTION)

        Xf_tr = X_finish.iloc[tr_i].copy()
        Xf_te = X_finish.iloc[te_i].copy()
        yf_tr = y_drop[tr_i]
        yf_te = y_drop[te_i]
        wf_tr = w_drop[tr_i]
        wf_te = w_drop[te_i]
    else:
        Xf_tr, Xf_te, yf_tr, yf_te, wf_tr, wf_te = train_test_split(
            X_finish, y_drop, w_drop,
            test_size=TEST_FRACTION,
            random_state=42,
            stratify=y_drop,
        )

    for c in ["pred_user_score10", "log1p_total_minutes"]:
        print(c, "in Xf_tr?", c in Xf_tr.columns, "in Xf_te?", c in Xf_te.columns)
        if c in Xf_tr.columns:
            print("  train std:", float(np.std(Xf_tr[c].to_numpy(dtype=float))))
        if c in Xf_te.columns:
            print("  test  std:", float(np.std(Xf_te[c].to_numpy(dtype=float))))

    tag_cols_f = [c for c in Xf_tr.columns if c.startswith("tag_")]
    genre_cols_f = [c for c in Xf_tr.columns if c.startswith("genre_")]

    tag_drop_means, tag_drop_counts, tag_drop_global = fit_multihot_rate_encoding(Xf_tr, yf_tr, tag_cols_f, smooth=50.0)
    gen_drop_means, gen_drop_counts, gen_drop_global = fit_multihot_rate_encoding(Xf_tr, yf_tr, genre_cols_f, smooth=20.0)

    Xf_tr = apply_multihot_rate_features(Xf_tr, tag_cols_f, tag_drop_means, tag_drop_counts, tag_drop_global, "tag_drop")
    Xf_te = apply_multihot_rate_features(Xf_te, tag_cols_f, tag_drop_means, tag_drop_counts, tag_drop_global, "tag_drop")

    Xf_tr = apply_multihot_rate_features(Xf_tr, genre_cols_f, gen_drop_means, gen_drop_counts, gen_drop_global, "genre_drop")
    Xf_te = apply_multihot_rate_features(Xf_te, genre_cols_f, gen_drop_means, gen_drop_counts, gen_drop_global, "genre_drop")

    Xf_te = Xf_te.reindex(columns=Xf_tr.columns, fill_value=0)

    dense_candidates_finish = [
        "episodes", "duration", "endDate_year",
        "log1p_popularity",  
        "meanScore_10", 
        "stats_log1p_n_total","stats_p_completed","stats_p_dropped","stats_retention",
        "stats_hype_gap","stats_p90plus","stats_p50minus",
         "log1p_total_minutes",

        "pred_user_score10",

        "tag_drop_mean_rate", "tag_drop_mean_support", "tag_drop_present",
        "genre_drop_mean_rate", "genre_drop_mean_support", "genre_drop_present",
    ]


    dense_num_finish, binary_num_finish = split_dense_vs_binary(Xf_tr, cat_cols_finish, dense_candidates_finish)

    prep_finish = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_finish),
        ("dense", "passthrough", dense_num_finish),
        ("bin", "passthrough", binary_num_finish),
    ], remainder="drop")

    pos = int((yf_tr == 1).sum())
    neg = int((yf_tr == 0).sum())

    pos = int((yf_tr == 1).sum())
    neg = int((yf_tr == 0).sum())

    pos_weight = min(np.sqrt(neg / max(pos, 1)), 8.0)

    sample_weight = wf_tr.copy()

    sample_weight[yf_tr == 1] *= pos_weight
    sample_weight = sample_weight.astype(np.float32)


    print(f"Drop model train balance: drop(pos)={pos} completed(neg)={neg}  pos_weight={pos_weight:.2f}")

    finish_base = fit_finish_xgb_with_early_stopping(
        Xf_tr, yf_tr, sample_weight,
        prep_finish=prep_finish,
        random_state=42,
        test_size=0.15,
        early_rounds=200,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(Xf_tr), dtype=np.float32)

    for tr_idx, va_idx in skf.split(Xf_tr, yf_tr):
        X_tr_i = Xf_tr.iloc[tr_idx]
        y_tr_i = yf_tr[tr_idx]
        w_tr_i = sample_weight[tr_idx]

        m = fit_finish_xgb_with_early_stopping(
            Xf_tr.iloc[tr_idx],
            yf_tr[tr_idx],
            sample_weight[tr_idx],
            prep_finish=prep_finish,
            random_state=42 + int(tr_idx[0]),
            test_size=0.15,
            early_rounds=200,
        )

        oof_proba[va_idx] = m.predict_proba(Xf_tr.iloc[va_idx])[:, 1].astype(np.float32)

    CAL_COLS = ["pred_user_score10", "log1p_total_minutes"]

    for c in CAL_COLS:
        if c not in Xf_tr.columns:
            Xf_tr[c] = 0.0
        if c not in Xf_te.columns:
            Xf_te[c] = 0.0

    calibrator = LogisticRegression(solver="lbfgs", max_iter=2000)

    cal_X = np.column_stack([
        oof_proba.astype(np.float32),
        *[Xf_tr[c].to_numpy(dtype=np.float32) for c in CAL_COLS]
    ])

    calibrator.fit(cal_X, yf_tr, sample_weight=sample_weight)

    def calibrated_proba(base_probs: np.ndarray, X_df: pd.DataFrame) -> np.ndarray:
        Xc = np.column_stack([
            base_probs.astype(np.float32),
            *[X_df[c].to_numpy(dtype=np.float32) for c in CAL_COLS]
        ])
        return calibrator.predict_proba(Xc)[:, 1]

    print("Stacker features = [base_prob] +", CAL_COLS)
    print("Stacker coef:", calibrator.coef_)
    print("Stacker intercept:", calibrator.intercept_)

    base_te = finish_base.predict_proba(Xf_te)[:, 1].astype(np.float32)

    ap_drop_base = average_precision_score(yf_te, base_te)
    print(f"AP_drop (base, uncalibrated): {ap_drop_base:.3f}")

    proba_drop = calibrated_proba(base_te, Xf_te).astype(np.float32)
    ap_drop = average_precision_score(yf_te, proba_drop)
    print(f"AP_drop (stacked): {ap_drop:.3f}")

    proba_finish = 1.0 - proba_drop
    ap_finish = average_precision_score(1 - yf_te, proba_finish)
    print(f"AP_finish: {ap_finish:.3f}")


    export_permutation_importance(
        finish_base,
        X=Xf_te,
        y=yf_te,
        out_csv="feature_importance_finish_perm.csv",
        top_k=60,
        title="Finish model (XGB; uncalibrated)",
        n_repeats=8,
        max_rows=2000,
        scoring="average_precision",
    )



    ths = np.linspace(0.01, 0.99, 199)

    best_t = 0.5
    best_ba = -1.0

    for t in ths:
        pred_t = (proba_drop >= t).astype(int)
        ba = balanced_accuracy_score(yf_te, pred_t)
        if ba > best_ba:
            best_ba = ba
            best_t = t

    pred_ba = np.asarray(proba_drop >= best_t, dtype=int)

    MIN_FINISH_RECALL = 0.80

    best_t_finish_safe = 0.50
    best_drop_recall = -1.0
    best_finish_recall = -1.0

    for t in ths:
        pred_t = (proba_drop >= t).astype(int)

        r_finish = recall_score(yf_te, pred_t, pos_label=0, zero_division=0)
        if r_finish < MIN_FINISH_RECALL:
            continue

        r_drop = recall_score(yf_te, pred_t, pos_label=1, zero_division=0)
        if r_drop > best_drop_recall:
            best_drop_recall = r_drop
            best_finish_recall = r_finish
            best_t_finish_safe = t

    pred_safe = np.asarray(proba_drop >= best_t_finish_safe, dtype=int)

    print(f"\nThreshold A (balanced accuracy): t={best_t:.3f}  BA={best_ba:.3f}")
    print("Confusion matrix @ BA threshold (rows=true [finish,drop], cols=pred [finish,drop]):")
    print(confusion_matrix(yf_te, pred_ba))
    print(classification_report(yf_te, pred_ba, digits=3, zero_division=0))

    ba_safe = balanced_accuracy_score(yf_te, pred_safe)
    print(f"\nThreshold B (finish recall >= {MIN_FINISH_RECALL:.2f}): t={best_t_finish_safe:.3f}  "
        f"finish_recall={best_finish_recall:.3f}  drop_recall={best_drop_recall:.3f}  BA={ba_safe:.3f}")
    print("Confusion matrix @ finish-safe threshold:")
    print(confusion_matrix(yf_te, pred_safe))
    print(classification_report(yf_te, pred_safe, digits=3, zero_division=0))


    print("\nRefitting finish model on ALL labeled data...")

    X_finish_full = X_finish.copy()

    tag_cols_full   = [c for c in X_finish_full.columns if c.startswith("tag_")]
    genre_cols_full = [c for c in X_finish_full.columns if c.startswith("genre_")]

    tag_drop_means_all, tag_drop_counts_all, tag_drop_global_all = fit_multihot_rate_encoding(
        X_finish_full, y_drop, tag_cols_full, smooth=50.0
    )
    gen_drop_means_all, gen_drop_counts_all, gen_drop_global_all = fit_multihot_rate_encoding(
        X_finish_full, y_drop, genre_cols_full, smooth=20.0
    )

    X_finish_full = apply_multihot_rate_features(
        X_finish_full, tag_cols_full, tag_drop_means_all, tag_drop_counts_all, tag_drop_global_all, "tag_drop"
    )
    X_finish_full = apply_multihot_rate_features(
        X_finish_full, genre_cols_full, gen_drop_means_all, gen_drop_counts_all, gen_drop_global_all, "genre_drop"
    )

    X_all_finish = X_all_finish.copy()
    X_all_finish = apply_multihot_rate_features(
        X_all_finish, tag_cols_full, tag_drop_means_all, tag_drop_counts_all, tag_drop_global_all, "tag_drop"
    )
    X_all_finish = apply_multihot_rate_features(
        X_all_finish, genre_cols_full, gen_drop_means_all, gen_drop_counts_all, gen_drop_global_all, "genre_drop"
    )

    X_all_finish = X_all_finish.reindex(columns=X_finish_full.columns, fill_value=0)

    dense_candidates_finish_full = [
        "episodes", "duration", "endDate_year",
        "log1p_popularity",  
        "meanScore_10", 
        "stats_log1p_n_total","stats_p_completed","stats_p_dropped","stats_retention",
        "stats_hype_gap","stats_p90plus","stats_p50minus",
        
        "log1p_total_minutes",
        "pred_user_score10",
        "tag_drop_mean_rate", "tag_drop_mean_support", "tag_drop_present",
        "genre_drop_mean_rate", "genre_drop_mean_support", "genre_drop_present",
    ]

    dense_num_finish, binary_num_finish = split_dense_vs_binary(
        X_finish_full, cat_cols_finish, dense_candidates_finish_full
    )

    prep_finish_full = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_finish),
        ("dense", "passthrough", dense_num_finish),
        ("bin", "passthrough", binary_num_finish),
    ], remainder="drop")

    pos_full = int((y_drop == 1).sum())
    neg_full = int((y_drop == 0).sum())
    pos_weight_full = min(np.sqrt(neg_full / max(pos_full, 1)), 8.0)

    sample_weight_full = w_drop.copy().astype(np.float32)
    sample_weight_full[y_drop == 1] *= pos_weight_full

    finish_base_full = fit_finish_xgb_with_early_stopping(
        X_finish_full, y_drop, sample_weight_full,
        prep_finish=prep_finish_full,
        random_state=42,
        test_size=0.15,
        early_rounds=200,
    )

    CAL_COLS = ["pred_user_score10", "log1p_total_minutes"]

    for c in CAL_COLS:
        if c not in X_finish_full.columns:
            X_finish_full[c] = 0.0

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba_full = np.zeros(len(X_finish_full), dtype=np.float32)

    for tr_idx, va_idx in skf.split(X_finish_full, y_drop):
        m = fit_finish_xgb_with_early_stopping(
            X_finish_full.iloc[tr_idx], y_drop[tr_idx], sample_weight_full[tr_idx],
            prep_finish=prep_finish_full,
            random_state=42 + int(tr_idx[0]),
            test_size=0.15,
            early_rounds=200,
        )
        oof_proba_full[va_idx] = m.predict_proba(X_finish_full.iloc[va_idx])[:, 1].astype(np.float32)

    cal_X_full = np.column_stack([
        oof_proba_full.astype(np.float32),
        *[X_finish_full[c].to_numpy(dtype=np.float32) for c in CAL_COLS]
    ])

    calibrator_full = LogisticRegression(solver="lbfgs", max_iter=2000)
    calibrator_full.fit(cal_X_full, y_drop, sample_weight=sample_weight_full)
    ths = np.linspace(0.01, 0.99, 199)

    oof_drop_p_full = calibrator_full.predict_proba(cal_X_full)[:, 1].astype(np.float32)

    best_t_oof = 0.50
    best_ba_oof = -1.0
    for t in ths:
        pred_t = (oof_drop_p_full >= t).astype(int)
        ba = balanced_accuracy_score(y_drop, pred_t)
        if ba > best_ba_oof:
            best_ba_oof = ba
            best_t_oof = t

    MIN_FINISH_RECALL = 0.80
    best_t_safe_oof = 0.50
    best_drop_recall_oof = -1.0
    best_finish_recall_oof = -1.0

    for t in ths:
        pred_t = (oof_drop_p_full >= t).astype(int)
        r_finish = recall_score(y_drop, pred_t, pos_label=0, zero_division=0)
        if r_finish < MIN_FINISH_RECALL:
            continue
        r_drop = recall_score(y_drop, pred_t, pos_label=1, zero_division=0)
        if r_drop > best_drop_recall_oof:
            best_drop_recall_oof = r_drop
            best_finish_recall_oof = r_finish
            best_t_safe_oof = t

    print(f"\n[OOF FULL] Threshold A (BA): t={best_t_oof:.3f}  BA={best_ba_oof:.3f}")
    print(f"[OOF FULL] Threshold B (finish recall >= {MIN_FINISH_RECALL:.2f}): "
        f"t={best_t_safe_oof:.3f}  finish_recall={best_finish_recall_oof:.3f}  drop_recall={best_drop_recall_oof:.3f}")

    class _FinishPredictor:
        def __init__(self, base_model, calibrator_lr, cal_cols):
            self.base_model = base_model
            self.calibrator_lr = calibrator_lr
            self.cal_cols = cal_cols

        def predict_proba(self, X: pd.DataFrame):
            base = self.base_model.predict_proba(X)[:, 1].astype(np.float32)
            Xc = np.column_stack([
                base,
                *[X[c].to_numpy(dtype=np.float32) for c in self.cal_cols]
            ])
            cal = self.calibrator_lr.predict_proba(Xc)[:, 1]
            return np.vstack([1.0 - cal, cal]).T

    finish_model = _FinishPredictor(finish_base_full, calibrator_full, CAL_COLS)



    prep_finish = prep_finish_full



    print("\nPredicting for all AniList anime...")

    n = len(X_all_score)
    pred_residual = np.empty(n, dtype=np.float32)
    pred_drop_p = np.empty(n, dtype=np.float32)

    mean_score_10_all = (
        pd.to_numeric(anilist_valid["meanScore"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32) / 10.0
    )

    score_transform = prep_score.transform
    score_predict = xgb_score.predict
    finish_predict_proba = finish_model.predict_proba

    for start in tqdm(range(0, n, BATCH_SIZE), desc="Predicting"):
        end = min(start + BATCH_SIZE, n)

        Xs = X_all_score.iloc[start:end]
        Xf = X_all_finish.iloc[start:end]

        Xs_t = score_transform(Xs)
        pred_residual[start:end] = score_predict(Xs_t).astype(np.float32) - np.float32(resid_bias)

        pred_drop_p[start:end] = finish_predict_proba(Xf)[:, 1].astype(np.float32)

    pred_score_10 = mean_score_10_all + pred_residual
    pred_score_10 = np.clip(pred_score_10, 1.0, 10.0)
    pred_score_100_raw = np.round(pred_score_10 * 10.0, 1)
    pred_score_100 = np.round(calibrate_score100(pred_score_100_raw), 1)


    pred_drop_p = np.clip(pred_drop_p, 0.0, 1.0)
    pred_finish_p = 1.0 - pred_drop_p



    pop = pd.to_numeric(anilist_valid.get("popularity", 0), errors="coerce").fillna(0).values
    logp = np.log1p(np.clip(pop, 0, None))
    pop_unc = 1.0 / (1.0 + logp)

    tag_count = all_tag_df_score.sum(axis=1).values.astype(float)
    max_tag = float(tag_count.max()) if tag_count.max() > 0 else 1.0
    tag_unc = 1.0 - (tag_count / max_tag)

    train_dense = X_score_full[dense_num_score].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mu = train_dense.mean(axis=0).to_numpy(dtype=np.float32)
    sigma = train_dense.std(axis=0).replace(0, 1.0).to_numpy(dtype=np.float32)

    all_dense = X_all_score[dense_num_score].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    z = np.abs((all_dense - mu) / sigma)
    ood_unc = np.clip(z.mean(axis=1) / 3.0, 0.0, 1.0)

    unc = (0.50 * pop_unc) + (0.25 * tag_unc) + (0.25 * ood_unc)
    conf = [confidence_label(float(u)) for u in unc]

    final_score = pred_score_100 * (FINAL_SCORE_FLOOR + (1.0 - FINAL_SCORE_FLOOR) * pred_finish_p)


    try:
        print("\nComputing SHAP explanations for score model...")

        X_all_score_t = prep_score.transform(X_all_score)

        try:
            score_feature_names = prep_score.get_feature_names_out()
        except Exception:
            score_feature_names = np.array([f"f{i}" for i in range(X_all_score_t.shape[1])], dtype=object)

        def clean_name(n: str) -> str:
            n = n.replace("cat__", "").replace("dense__", "").replace("bin__", "")
            return n

        explainer = shap.TreeExplainer(xgb_score)
        shap_values = explainer.shap_values(X_all_score_t)

        def top_shap_signals(i: int, k: int = 6, max_priors: int = 1) -> str:
            v = shap_values[i]
            order = np.argsort(np.abs(v))[::-1]

            parts = []
            priors_used = 0

            for j in order:
                name = clean_name(str(score_feature_names[j]))

                base = name.split("=", 1)[0]

                is_prior = (base in GLOBAL_PRIORS)

                if base in EXPLANATION_BLACKLIST:
                    continue
                if is_prior and priors_used >= max_priors:
                    continue

                delta = float(v[j])
                sign = "+" if delta >= 0 else "-"
                parts.append(f"{sign}{name}")

                if is_prior:
                    priors_used += 1

                if len(parts) >= k:
                    break

            return "|".join(parts)


        top_score_signals = [top_shap_signals(i, k=8) for i in range(len(X_all_score))]

    except Exception as e:
        print(f"[WARN] SHAP Why_Score disabled (install shap?). Reason: {e}")
        top_score_signals = [""] * len(X_all_score)

    print("\nComputing per-anime finish explanations (batched ablation)...")

    finish_explain_cols = []
    for c in dense_num_finish:
        if c in X_all_finish.columns:
            finish_explain_cols.append(c)

    bin_freq = X_finish[binary_num_finish].sum(axis=0).sort_values(ascending=False)
    finish_explain_cols += [c for c in bin_freq.head(30).index if c in X_all_finish.columns]

    finish_explain_cols = list(dict.fromkeys(finish_explain_cols))

    base_p_all = finish_model.predict_proba(X_all_finish)[:, 1].astype(np.float32)

    deltas = np.zeros((len(X_all_finish), len(finish_explain_cols)), dtype=np.float32)

    for j, col in enumerate(tqdm(finish_explain_cols, desc="Finish ablation (by col)")):
        Xp = X_all_finish.copy()
        Xp[col] = 0
        p2 = finish_model.predict_proba(Xp)[:, 1].astype(np.float32)
        deltas[:, j] = base_p_all - p2

    def pick_top_signals_from_deltas(
        row_deltas: np.ndarray,
        cols: list[str],
        k: int = 6,
        max_stats: int = 2,
        max_priors: int = 1,
    ) -> str:
        order = np.argsort(np.abs(row_deltas))[::-1]

        parts = []
        stats_used = 0
        priors_used = 0

        for j in order:
            col = cols[j]
            t = _signal_type(col)

            if col in EXPLANATION_BLACKLIST:
                continue
            if t == "stats" and stats_used >= max_stats:
                continue
            if t == "prior" and priors_used >= max_priors:
                continue

            delta = float(row_deltas[j])
            sign = "+" if delta >= 0 else "-"
            parts.append(f"{sign}{col}")

            if t == "stats":
                stats_used += 1
            elif t == "prior":
                priors_used += 1

            if len(parts) >= k:
                break

        return "|".join(parts)

    k = 8
    top_finish_signals = []
    for i in range(len(X_all_finish)):
        top_finish_signals.append(
            pick_top_signals_from_deltas(
                deltas[i],
                finish_explain_cols,
                k=k,
                max_stats=1,
                max_priors=1,
            )
        )

    your_score_map = (
        personal[["AniList ID", "Your_Score"]]
        .groupby("AniList ID")["Your_Score"]
        .max()
    )
    your_status_map = (
        personal[["AniList ID", "Status"]]
        .dropna(subset=["AniList ID"])
        .assign(Status=lambda d: d["Status"].astype(str).str.upper())
        .groupby("AniList ID")["Status"]
        .agg(lambda s: s.iloc[-1])
    )

    out = anilist_valid[[
        "id", "idMal", "title_romaji", "title_english","startDate_year",
        "format", "popularity", "meanScore"
    ]].copy()

    out["Score_Split"] = out["id"].astype(int).map(
        lambda aid: "score_train" if aid in score_train_ids
        else ("score_test" if aid in score_test_ids else "not_in_score_set")
    )
    out["Your_Status"] = out["id"].map(your_status_map)
    out["Your_Score"] = out["id"].map(your_score_map)
    out["Predicted_Score_Raw"] = pred_score_100_raw
    out["Predicted_Score"] = pred_score_100

    out["P_finish"] = np.round(pred_finish_p, 3)
    out["FinalScore"] = np.round(final_score, 2)
    out["Pred_Drop_Label"] = (pred_drop_p >= best_t_finish_safe).astype(int)
    out["Pred_Finish_Label"] = 1 - out["Pred_Drop_Label"]


    score_imp_df = None
    try:
        score_imp_df = pd.read_csv("feature_importance_score_perm.csv")
    except Exception:
        score_imp_df = None

    finish_imp_df = None
    try:
        finish_imp_df = pd.read_csv("feature_importance_finish_perm.csv")
    except Exception:
        finish_imp_df = None

    finish_threshold_used = 1.0 - float(best_t_finish_safe)

    id_to_idx = anilist_pos
    why_score_col = []
    why_finish_col = []

    q20, q40, q60, q80, q95 = make_user_score_quantiles(personal)
    print(f"User score quantiles: q20={q20:.2f}, q40={q40:.2f}, q60={q60:.2f}, q80={q80:.2f}, q95={q95:.2f}")

    for _, r in out.iterrows():
        aid = int(r["id"])
        idx = id_to_idx.get(aid, None)
        if idx is None:
            why_score_col.append("")
            why_finish_col.append("")
            continue

        xs_row = X_all_score.iloc[idx]
        xf_row = X_all_finish.iloc[idx]

        why_s = build_why_score(idx, r, xs_row, score_imp_df, q20, q40, q60, q80, q95)
        why_f = build_why_finish(idx, r, xf_row, finish_imp_df, finish_threshold_used)

        why_score_col.append(why_s)
        why_finish_col.append(why_f)

    out["Why_Score"] = why_score_col
    out["Why_Finish"] = why_finish_col
    out["Top_Score_Signals"] = top_score_signals
    out["Top_Finish_Signals"] = top_finish_signals
    out["Confidence"] = conf

    out["Prediction_Error"] = out["Predicted_Score"] - out["Your_Score"]
    out["Abs_Error"] = out["Prediction_Error"].abs()

    out = out.sort_values("FinalScore", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
