"""
betas.py

Fit a linear regression model for the AniList Overrated Index (OI) baseline.

This script uses AniList community status and score-distribution data to learn
global regression coefficients that estimate an anime's expected rating from
engagement patterns.

The fitted model uses:
- completion share
- in-progress / dropped share
- score-distribution proportions
- log popularity

Input:
- anilist_anime_data_complete.csv

Output:
- OI_betas.json

Notes:
- The model is fit without a separate intercept term because the feature vector
  includes a constant 1 column
- The exported coefficients can be reused later to compute expected scores and OI
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression

df = pd.read_csv('anilist_anime_data_complete.csv')

X_list = []
y_list = []

status_map = {
    "CURRENT": "watching",
    "COMPLETED": "completed",
    "PLANNING": "plan_to_watch",
    "DROPPED": "dropped",
    "ON_HOLD": "on_hold"
}

for idx, row in df.iterrows():
    try:
        status_json = json.loads(row['stats_statusDistribution'])
        score_json = json.loads(row['stats_scoreDistribution'])
    except:
        continue

    status = {status_map.get(item['status'], item['status'].lower()): item.get('amount', 0)
              for item in status_json}

    completed = status.get('completed', 0)
    watching = status.get('watching', 0)
    on_hold = status.get('on_hold', 0)
    dropped = status.get('dropped', 0)
    plan_to_watch = status.get('plan_to_watch', 0)

    total_members = completed + watching + on_hold + dropped + plan_to_watch
    if total_members == 0:
        continue

    scores = {}
    for item in score_json:
        score_1to10 = int(item['score'] / 10)
        if score_1to10 < 1:
            score_1to10 = 1
        scores[str(score_1to10)] = scores.get(str(score_1to10), 0) + item.get('amount', 0)

    total_score_votes = sum(scores.values())
    if total_score_votes == 0:
        continue

    C_eff = completed / total_members
    D = (watching + on_hold + dropped) / total_members
    S = np.array([scores.get(str(i), 0)/total_score_votes for i in range(1,11)])
    log_pop = np.log(total_members)

    feature_vector = [1, C_eff, C_eff**2, D, D**2] + list(S) + [log_pop]

    X_list.append(feature_vector)

    R_obs = sum(S * np.arange(1,11))
    y_list.append(R_obs)

X = np.array(X_list)
y = np.array(y_list)

reg = LinearRegression(fit_intercept=False)
reg.fit(X, y)

feature_names = ['intercept','C_eff','C_eff^2','D','D^2'] + [f'S{i}' for i in range(1,11)] + ['log_pop']
betas = dict(zip(feature_names, reg.coef_))

print("Calculated Beta Coefficients:")
for k, v in betas.items():
    print(f"{k}: {v:.4f}")

import json
with open('OI_betas.json', 'w') as f:
    json.dump(betas, f, indent=4)
