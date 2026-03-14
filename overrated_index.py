"""
overrated_index.py

Estimate an "Overrated Index" (OI) for anime by comparing observed AniList mean
scores against scores expected from engagement and metadata features.

This script:
- builds per-anime regression features from AniList status-distribution data
- fits a separate linear model for each release year
- predicts an expected score for each anime
- computes OI as observed score minus expected score
- labels each title as Overrated, Underrated, or Fairly rated

Input:
- anilist_anime_data_complete.csv

Outputs:
- OI_yearly_betas_new.json
- anime_OI_meaningful.csv
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import json

df = pd.read_csv('anilist_anime_data_complete.csv')

status_map = {
    "CURRENT": "watching",
    "COMPLETED": "completed",
    "PLANNING": "plan_to_watch",
    "DROPPED": "dropped",
    "ON_HOLD": "on_hold"
}

source_types = {s:i for i,s in enumerate(df['source'].dropna().unique())}
format_types = {f:i for i,f in enumerate(df['format'].dropna().unique())}

def build_features(row):
    try:
        status_json = json.loads(row['stats_statusDistribution'])
    except:
        return None, None, None

    status = {status_map.get(item['status'], item['status'].lower()): item.get('amount',0)
              for item in status_json}

    completed = status.get('completed',0)
    watching = status.get('watching',0)
    on_hold = status.get('on_hold',0)
    dropped = status.get('dropped',0)
    plan_to_watch = status.get('plan_to_watch',0)
    total_members = completed + watching + on_hold + dropped + plan_to_watch
    if total_members == 0:
        return None, None, None

    C_eff = completed / total_members
    D = (watching + on_hold + dropped) / total_members
    log_pop = np.log(total_members)
    episodes = row['episodes'] if not pd.isna(row['episodes']) else 0
    duration = row['duration'] if not pd.isna(row['duration']) else 0
    source_encoded = source_types.get(row['source'], -1)
    format_encoded = format_types.get(row['format'], -1)

    X = [1, C_eff, C_eff**2, D, D**2, log_pop, episodes, duration, source_encoded, format_encoded]

    R_obs = row['meanScore'] / 10 if not pd.isna(row['meanScore']) else None
    if R_obs is None:
        return None, None, None

    return X, R_obs, row['startDate_year']

X_year = {}
y_year = {}
print("Building features for regression...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    X_feat, R_obs, year = build_features(row)
    if X_feat is None:
        continue
    if year not in X_year:
        X_year[year] = []
        y_year[year] = []
    X_year[year].append(X_feat)
    y_year[year].append(R_obs)

print("Fitting regression for each year...")
yearly_betas = {}
for year in tqdm(X_year.keys()):
    X_arr = np.array(X_year[year])
    y_arr = np.array(y_year[year])
    if len(y_arr) < 5:
        continue
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_arr, y_arr)
    feature_names = ['intercept','C_eff','C_eff^2','D','D^2','log_pop','episodes','duration','source','format']
    yearly_betas[year] = dict(zip(feature_names, reg.coef_))

with open('OI_yearly_betas_new.json','w') as f:
    json.dump(yearly_betas, f, indent=4)
print("Saved yearly betas to OI_yearly_betas_new.json")

ois = []
R_expected_list = []

print("Calculating OI for all anime...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    X_feat, R_obs, year = build_features(row)
    if X_feat is None or year not in yearly_betas:
        ois.append(None)
        R_expected_list.append(None)
        continue
    beta_vector = np.array([yearly_betas[year][k] for k in ['intercept','C_eff','C_eff^2','D','D^2','log_pop','episodes','duration','source','format']])
    R_expected_10 = np.dot(X_feat, beta_vector)
    R_expected = R_expected_10 * 10
    OI = (R_obs*10) - R_expected
    ois.append(OI)
    R_expected_list.append(R_expected)

df['OI'] = ois
df['R_expected'] = R_expected_list

def interpret_oi(oi):
    if oi is None:
        return None
    if oi > 3:
        return "Overrated"
    elif oi < -3:
        return "Underrated"
    else:
        return "Fairly rated"

df['Interpretation'] = df['OI'].apply(interpret_oi)

df_output = df[['title_romaji','title_english','meanScore','R_expected','OI','Interpretation']].dropna()
df_output = df_output.sort_values(by='OI', ascending=False).reset_index(drop=True)
df_output.to_csv('anime_OI_meaningful.csv', index=False)
print("Saved final OI table as anime_OI_meaningful.csv")
