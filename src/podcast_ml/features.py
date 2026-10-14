"""Feature engineering + preprocessing (fit once, reuse for training and serving)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Listening_Time_minutes"
CAT_COLS = ["Podcast_Name", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]
RAW_COLS = ["Podcast_Name", "Episode_Title", "Episode_Length_minutes", "Genre",
            "Host_Popularity_percentage", "Publication_Day", "Publication_Time",
            "Guest_Popularity_percentage", "Number_of_Ads", "Episode_Sentiment"]
SENT_SCORE = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}
WEEKEND = {"Saturday", "Sunday"}

NUM_FEATURES = [
    "length", "log_length", "host_pop", "guest_pop", "ads", "episode_no",
    "host_x_length", "guest_x_length", "ads_per_hour", "is_weekend", "sentiment_score",
    "length_missing", "guest_missing", "ads_missing", "host_missing",
]


class PodcastPreprocessor:
    """Turns the raw Kaggle-style dataframe into the dict of arrays the Keras model eats."""

    def __init__(self) -> None:
        self.vocab: dict[str, dict[str, int]] = {}
        self.medians: dict[str, float] = {}
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.y_mean = 0.0
        self.y_std = 1.0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _episode_number(df: pd.DataFrame) -> pd.Series:
        if "Episode_Number" in df:
            return pd.to_numeric(df["Episode_Number"], errors="coerce")
        return pd.to_numeric(df["Episode_Title"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    def _numeric_frame(self, df: pd.DataFrame, fitting: bool = False) -> pd.DataFrame:
        length = pd.to_numeric(df["Episode_Length_minutes"], errors="coerce")
        host = pd.to_numeric(df["Host_Popularity_percentage"], errors="coerce")
        guest = pd.to_numeric(df["Guest_Popularity_percentage"], errors="coerce")
        ads = pd.to_numeric(df["Number_of_Ads"], errors="coerce")
        epno = self._episode_number(df)
        if fitting:
            self.medians = {"length": float(length.median()), "host": float(host.median()),
                            "guest": float(guest.median()), "ads": float(ads.median()),
                            "episode": float(epno.median())}
        m = self.medians
        out = pd.DataFrame(index=df.index)
        out["length_missing"] = length.isna().astype(float)
        out["guest_missing"] = guest.isna().astype(float)
        out["ads_missing"] = ads.isna().astype(float)
        out["host_missing"] = host.isna().astype(float)
        length = length.fillna(m["length"]).clip(1, 300)
        host = host.fillna(m["host"]).clip(0, 120)
        guest = guest.fillna(m["guest"]).clip(0, 120)
        ads = ads.fillna(m["ads"]).clip(0, 10)
        epno = epno.fillna(m["episode"]).clip(0, 500)
        out["length"] = length
        out["log_length"] = np.log1p(length)
        out["host_pop"] = host
        out["guest_pop"] = guest
        out["ads"] = ads
        out["episode_no"] = epno
        out["host_x_length"] = host * length / 100.0
        out["guest_x_length"] = guest * length / 100.0 * (1 - out["guest_missing"])
        out["ads_per_hour"] = ads / (length / 60.0)
        out["is_weekend"] = df["Publication_Day"].isin(WEEKEND).astype(float)
        out["sentiment_score"] = df["Episode_Sentiment"].map(SENT_SCORE).fillna(0.0)
        return out[NUM_FEATURES]

    # --------------------------------------------------------------------- API
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "PodcastPreprocessor":
        for c in CAT_COLS:
            cats = sorted(df[c].dropna().astype(str).unique())
            self.vocab[c] = {v: i + 1 for i, v in enumerate(cats)}  # 0 = unknown / missing
        num = self._numeric_frame(df, fitting=True)
        self.mean_ = num.mean().to_numpy(dtype="float64")
        self.std_ = num.std().replace(0, 1).to_numpy(dtype="float64")
        if y is not None:
            self.y_mean, self.y_std = float(y.mean()), float(y.std())
        return self

    def vocab_size(self, col: str) -> int:
        return len(self.vocab[col]) + 1

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for c in CAT_COLS:
            out[f"cat_{c}"] = df[c].astype(str).map(self.vocab[c]).fillna(0).astype("int32").to_numpy().reshape(-1, 1)
        num = self._numeric_frame(df).to_numpy(dtype="float64")
        out["num"] = ((num - self.mean_) / self.std_).astype("float32")
        return out

    def scale_y(self, y):
        return ((np.asarray(y, dtype="float64") - self.y_mean) / self.y_std).astype("float32")

    def unscale_y(self, y):
        return np.asarray(y, dtype="float64") * self.y_std + self.y_mean
