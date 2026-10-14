"""Inference wrapper used by the Streamlit app (and anything else)."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import keras
import numpy as np
import pandas as pd

from .features import RAW_COLS
from .model import mc_dropout_predict, predict_scaled
from .train import physical_limits

ROOT = Path(__file__).resolve().parents[2]


class PodcastPredictor:
    def __init__(self, model_dir: Path | str = ROOT / "models") -> None:
        d = Path(model_dir)
        self.prep = joblib.load(d / "preprocessor.joblib")
        self.model = keras.models.load_model(d / "podcast_model.keras", compile=False)
        self.calib = json.loads((d / "calibration.json").read_text())
        self.podcasts = list(self.prep.vocab["Podcast_Name"])
        self.genres = list(self.prep.vocab["Genre"])
        self.median_length = self.prep.medians["length"]

    def _prep_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Episode_Title" not in df:
            df["Episode_Title"] = "Episode " + df.get("Episode_Number", 50).astype(str)
        return df[RAW_COLS]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = self._prep_df(df)
        raw = self.prep.unscale_y(predict_scaled(self.model, self.prep.transform(df)))
        return physical_limits(raw, df["Episode_Length_minutes"])

    def predict_with_interval(self, df: pd.DataFrame, n_samples: int = 30) -> pd.DataFrame:
        """Point prediction, 90% interval (calibrated on validation residuals) and an MC-dropout
        'disagreement' score describing how unsure the network itself is."""
        df = self._prep_df(df)
        x = self.prep.transform(df)
        length = pd.to_numeric(df["Episode_Length_minutes"], errors="coerce")
        len_fill = length.fillna(self.median_length).to_numpy(dtype="float64")
        mean = physical_limits(self.prep.unscale_y(predict_scaled(self.model, x)), length)
        mc = self.prep.unscale_y(mc_dropout_predict(self.model, x, n_samples))
        lo = physical_limits(mean + self.calib["q_lo"] * len_fill, length)
        hi = physical_limits(mean + self.calib["q_hi"] * len_fill, length)
        return pd.DataFrame({"prediction": mean, "low": np.minimum(lo, mean), "high": np.maximum(hi, mean),
                             "mc_std": mc.std(axis=0), "episode_length": len_fill,
                             "length_known": length.notna().to_numpy()}, index=df.index)
