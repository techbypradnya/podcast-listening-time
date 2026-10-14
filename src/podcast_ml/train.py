"""Training pipeline (used by the CLI *and* the notebook)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .features import RAW_COLS, TARGET, PodcastPreprocessor

ROOT = Path(__file__).resolve().parents[2]


def find_dataset(root: Path = ROOT) -> Path:
    """Prefer the real Kaggle file when present, else the bundled synthetic one."""
    for p in (root / "data" / "kaggle" / "train.csv", root / "data" / "train.csv"):
        if p.exists():
            return p
    raise FileNotFoundError("Run `python data/make_synthetic_data.py` or add data/kaggle/train.csv")


def split_data(df: pd.DataFrame, seed: int = 42, val: float = 0.15, test: float = 0.15):
    train_df, test_df = train_test_split(df, test_size=test, random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=val / (1 - test), random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def regression_metrics(y_true, y_pred) -> dict:
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))}


def physical_limits(pred: np.ndarray, length: pd.Series) -> np.ndarray:
    """A person cannot listen for a negative time, or longer than the episode."""
    pred = np.maximum(pred, 0.0)
    length = pd.to_numeric(length, errors="coerce").to_numpy(dtype="float64")
    return np.where(np.isnan(length), pred, np.minimum(pred, length))


def fit_model(train_df, val_df, epochs: int = 60, batch_size: int = 512, seed: int = 42,
              verbose: int = 2, **model_kw):
    import keras
    from .model import build_model

    keras.utils.set_random_seed(seed)
    prep = PodcastPreprocessor().fit(train_df[RAW_COLS], train_df[TARGET])
    xtr, xva = prep.transform(train_df[RAW_COLS]), prep.transform(val_df[RAW_COLS])
    ytr, yva = prep.scale_y(train_df[TARGET]), prep.scale_y(val_df[TARGET])
    model = build_model(prep, **model_kw)
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)]
    hist = model.fit(xtr, ytr, validation_data=(xva, yva), epochs=epochs, batch_size=batch_size,
                     callbacks=cbs, verbose=verbose)
    return model, prep, hist.history


def predict_minutes(model, prep, df: pd.DataFrame) -> np.ndarray:
    from .model import predict_scaled
    raw = prep.unscale_y(predict_scaled(model, prep.transform(df[RAW_COLS])))
    return physical_limits(raw, df["Episode_Length_minutes"])


def interval_calibration(y_true, y_pred, length_filled, level: float = 0.9) -> dict:
    """Split-conformal style: quantiles of the *relative* residual (resid / episode length)."""
    rel = (np.asarray(y_true) - np.asarray(y_pred)) / np.asarray(length_filled)
    a = (1 - level) / 2
    return {"level": level, "q_lo": float(np.quantile(rel, a)), "q_hi": float(np.quantile(rel, 1 - a))}


def save_artifacts(out_dir: Path, model, prep, extras: dict) -> None:
    import joblib
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "podcast_model.keras")
    joblib.dump(prep, out_dir / "preprocessor.joblib")
    for name, obj in extras.items():
        (out_dir / f"{name}.json").write_text(json.dumps(obj, indent=2))


def main() -> None:
    from .explain import permutation_importance
    ap = argparse.ArgumentParser(description="Train the podcast listening-time model")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", type=Path, default=ROOT / "models")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = pd.read_csv(a.data or find_dataset()).dropna(subset=[TARGET])
    tr, va, te = split_data(df, a.seed)
    model, prep, hist = fit_model(tr, va, epochs=a.epochs, seed=a.seed)
    pred_va, pred_te = predict_minutes(model, prep, va), predict_minutes(model, prep, te)
    metrics = {"test": regression_metrics(te[TARGET], pred_te), "val": regression_metrics(va[TARGET], pred_va),
               "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    print(json.dumps(metrics, indent=2))
    len_fill = va["Episode_Length_minutes"].fillna(prep.medians["length"])
    calib = interval_calibration(va[TARGET], pred_va, len_fill)
    imp = permutation_importance(model, prep, te, te[TARGET])
    save_artifacts(a.out, model, prep, {"metrics": metrics, "history": hist, "calibration": calib,
                                        "feature_importance": imp})
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
