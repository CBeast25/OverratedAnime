# AniList Recommendation + Overrated Index

I built this because my plan-to-watch list got way too big, and I kept running into the same problem:

- I don’t know what I’ll actually enjoy  
- and I don’t know what I’ll realistically finish  

This uses my AniList data plus global AniList stats to try to answer both.

It predicts:

- how much I’ll like something  
- whether I’ll actually finish it  

---

## What it does

### Recommendations

Takes everything and ranks anime based on:

- predicted score  
- probability I’ll finish it  

These get combined into a final score, with a few adjustments (like not overloading on the same franchise or occasionally surfacing less popular titles).

At the end, this produces a ranked list of what to watch next.

---

### Overrated Index (OI)

This estimates what an anime *should* be rated based on:

- completion rates  
- drop rates  
- score distribution  
- popularity  

Then compares that to the actual score.

Outputs:

- overrated  
- underrated  
- fairly rated  

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Data

Pull the AniList dataset:

```bash
python fetch_data.py
```

Reduce it to a smaller working dataset:

```bash
python reduce_anilist_csv.py --in data/raw/anilist_anime_data_complete.csv --out anilist_anime_reduced.csv
```

---

## Training

```bash
python score_predictor.py
```

This generates:

- `anilist_personalized_predictions_full.csv`

---

## Generating Recommendations

```bash
python recommend_next.py
```

Outputs:

- a clean recommendation table  
- a raw version for deeper inspection  

---

## Cleaning plan-to-watch

```bash
python drop_planning_candidates.py
```

Helps identify entries that are unlikely to be worth keeping.

---

## Overrated Index

```bash
python betas.py
python overrated_index.py
```

Outputs a table comparing expected vs actual scores along with the OI value.

---

## Example output

| Title     | Pred | P_finish | Final |
|-----------|------|----------|-------|
| Monster   | 86.5 | 0.91     | 78.7  |
| Ping Pong | 83.2 | 0.88     | 73.2  |

---

## How I use this

- run the model periodically  
- check the top ~20 recommendations  
- trim my plan-to-watch list  
- occasionally use OI to find underrated titles  

---

## Notes

- large datasets are not included  
- requires your own AniList export  
- outputs are primarily CSV files  

---

## Future improvements

- better handling of tags and genres  
- improved visualization (beyond CSV outputs)  
