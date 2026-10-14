"""Generate a synthetic dataset that mirrors the schema of the Kaggle
Playground Series S5E4 "Predict Podcast Listening Time" competition.

Real data: https://www.kaggle.com/competitions/playground-series-s5e4
If you download the real train.csv, put it in data/kaggle/train.csv and the
notebook / training script will automatically prefer it.

Usage:  python data/make_synthetic_data.py --rows 75000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# podcast name -> home genre
PODCASTS = {
    "Tech Trends": "Technology", "Gadget Geeks": "Technology", "Code Corner": "Technology",
    "Future Forward": "Technology", "Byte Size": "Technology", "Silicon Stories": "Technology",
    "Money Matters": "Business", "Business Insights": "Business", "Startup Stories": "Business",
    "Market Watch": "Business", "Career Compass": "Business",
    "Study Sessions": "Education", "Education Station": "Education", "Learning Lab": "Education",
    "Brain Boost": "Education", "History Hour": "Education",
    "Laugh Line": "Comedy", "Joke Junction": "Comedy", "Comedy Corner": "Comedy",
    "Funny Folks": "Comedy", "Giggle Garage": "Comedy",
    "Health Hour": "Health", "Wellness Wave": "Health", "Fitness First": "Health",
    "Mind & Body": "Health", "Healthy Living": "Health", "Sleep Society": "Health",
    "Lifestyle Lounge": "Lifestyle", "Style Guide": "Lifestyle", "Home Harmony": "Lifestyle",
    "Life Lessons": "Lifestyle", "Fashion Forward": "Lifestyle",
    "Melody Mix": "Music", "Tune Town": "Music", "Music Matters": "Music", "Vinyl Vibes": "Music",
    "Daily Digest": "News", "News Roundup": "News", "Current Affairs": "News", "World Watch": "News",
    "Sport Spot": "Sports", "Game Day": "Sports", "Sports Weekly": "Sports", "Pitch Perfect": "Sports",
    "Mystery Matters": "True Crime", "True Crime Tonight": "True Crime", "Cold Case Files": "True Crime",
    "Unsolved Tales": "True Crime", "Dark Corners": "True Crime",
}
GENRES = sorted(set(PODCASTS.values()))
GENRE_EFFECT = {"True Crime": .045, "Comedy": .02, "Health": .01, "Technology": 0.0, "Sports": 0.0,
                "Business": -.01, "Lifestyle": -.012, "Music": -.02, "Education": -.005, "News": -.03}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_EFFECT = dict(zip(DAYS, [-.012, -.006, 0, .004, .002, .018, .02]))
TIMES = ["Morning", "Afternoon", "Evening", "Night"]
TIME_EFFECT = {"Morning": .0, "Afternoon": -.015, "Evening": .02, "Night": .01}
SENT_EFFECT = {"Positive": .03, "Neutral": 0.0, "Negative": -.04}


def generate(n: int, seed: int = 42, with_target: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = np.array(list(PODCASTS))
    podcast_quality = dict(zip(names, np.random.default_rng(7).normal(0, .03, len(names))))
    pod = rng.choice(names, n)
    home = np.array([PODCASTS[p] for p in pod])
    genre = np.where(rng.random(n) < .7, home, rng.choice(GENRES, n))
    length = np.clip(rng.normal(65, 30, n), 5, 120).round(2)
    host = np.clip(rng.normal(59, 23, n), 1, 119).round(2)
    guest = np.clip(rng.normal(53, 28, n), 0, 119).round(2)
    ads = rng.choice([0, 1, 2, 3], n, p=[.28, .27, .27, .18]).astype(float)
    day = rng.choice(DAYS, n)
    time = rng.choice(TIMES, n)
    sent = rng.choice(list(SENT_EFFECT), n, p=[.33, .34, .33])
    ep_no = rng.integers(1, 101, n)

    df = pd.DataFrame({
        "id": np.arange(n), "Podcast_Name": pod, "Episode_Title": [f"Episode {i}" for i in ep_no],
        "Episode_Length_minutes": length, "Genre": genre, "Host_Popularity_percentage": host,
        "Publication_Day": day, "Publication_Time": time, "Guest_Popularity_percentage": guest,
        "Number_of_Ads": ads, "Episode_Sentiment": sent})

    has_guest = rng.random(n) > .19
    if with_target:
        ads_per_hour = ads / (length / 60)
        ratio = (0.70
                 + 0.0014 * (host - 59)
                 + np.where(has_guest, 0.0009 * (guest - 53), 0.0)
                 + np.array([SENT_EFFECT[s] for s in sent])
                 + np.array([GENRE_EFFECT[g] for g in genre])
                 + np.array([DAY_EFFECT[d] for d in day])
                 + np.array([TIME_EFFECT[t] for t in time])
                 - 0.018 * ads - 0.012 * ads_per_hour
                 - 0.0007 * (length - 65)
                 + np.array([podcast_quality[p] for p in pod])
                 + rng.normal(0, .055, n))
        df["Listening_Time_minutes"] = np.clip(length * ratio, 0, length).round(5)
    df.loc[~has_guest, "Guest_Popularity_percentage"] = np.nan

    # realistic missingness
    df.loc[rng.random(n) < .116, "Episode_Length_minutes"] = np.nan
    df.loc[rng.random(n) < .02, "Host_Popularity_percentage"] = np.nan
    df.loc[rng.random(n) < .002, "Number_of_Ads"] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=75000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    generate(a.rows, a.seed).to_csv(HERE / "train.csv", index=False)
    generate(500, a.seed + 1, with_target=False).to_csv(HERE / "sample_batch.csv", index=False)
    print(f"wrote {a.rows} rows -> {HERE / 'train.csv'} and sample_batch.csv")
