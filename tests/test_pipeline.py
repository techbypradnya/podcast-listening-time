import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from make_synthetic_data import generate  # noqa: E402
from podcast_ml.features import NUM_FEATURES, RAW_COLS, TARGET, PodcastPreprocessor  # noqa: E402
from podcast_ml.train import physical_limits  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return generate(2000, seed=1)


def test_preprocessor_shapes_and_no_nans(df):
    prep = PodcastPreprocessor().fit(df[RAW_COLS], df[TARGET])
    x = prep.transform(df[RAW_COLS])
    assert x["num"].shape == (len(df), len(NUM_FEATURES))
    assert not np.isnan(x["num"]).any()
    assert all(v.shape == (len(df), 1) for k, v in x.items() if k.startswith("cat_"))


def test_unseen_category_maps_to_zero(df):
    prep = PodcastPreprocessor().fit(df[RAW_COLS], df[TARGET])
    new = df[RAW_COLS].head(3).copy()
    new["Podcast_Name"] = "Never Seen Before"
    assert (prep.transform(new)["cat_Podcast_Name"] == 0).all()


def test_target_scaling_roundtrip(df):
    prep = PodcastPreprocessor().fit(df[RAW_COLS], df[TARGET])
    y = df[TARGET].head(20)
    assert np.allclose(prep.unscale_y(prep.scale_y(y)), y, atol=1e-4)


def test_physical_limits():
    out = physical_limits(np.array([-5.0, 30.0, 90.0, 50.0]), pd.Series([60, 60, 60, np.nan]))
    assert out.tolist() == [0.0, 30.0, 60.0, 50.0]


def test_predictor_end_to_end():
    if not (ROOT / "models" / "podcast_model.keras").exists():
        pytest.skip("train the model first")
    from podcast_ml.predict import PodcastPredictor
    p = PodcastPredictor(ROOT / "models")
    d = generate(50, seed=3, with_target=False)
    out = p.predict_with_interval(d, n_samples=3)
    assert len(out) == 50
    assert (out.low <= out.prediction + 1e-6).all() and (out.prediction <= out.high + 1e-6).all()
    assert (out.prediction >= 0).all()
