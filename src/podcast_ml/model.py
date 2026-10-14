"""Wide & Deep neural network with entity embeddings and residual blocks (Keras 3)."""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
from keras import layers

from .features import CAT_COLS, NUM_FEATURES, PodcastPreprocessor


def embedding_dim(n_categories: int) -> int:
    return int(min(16, max(3, round(1.6 * n_categories ** 0.56))))


def build_model(prep: PodcastPreprocessor, hidden=(256, 128, 64), dropout: float = 0.15,
                lr: float = 2e-3, l2: float = 1e-5, weight_decay: float = 1e-4) -> keras.Model:
    """Architecture
    -------------
    * one embedding per categorical column  (learns podcast / genre / time "personalities")
    * standardised numeric block (15 engineered features)
    * DEEP tower : concat -> N x [Dense -> LayerNorm -> Swish -> Dropout] with skip connections
    * WIDE tower : a single linear layer on the numeric block (captures the ~linear length effect)
    * output     : deep + wide  (standardised minutes)
    """
    reg = keras.regularizers.l2(l2)
    cat_inputs, cat_vecs = [], []
    for c in CAT_COLS:
        inp = layers.Input(shape=(1,), dtype="int32", name=f"cat_{c}")
        n = prep.vocab_size(c)
        e = layers.Embedding(n, embedding_dim(n), name=f"emb_{c}")(inp)
        cat_inputs.append(inp)
        cat_vecs.append(layers.Flatten()(e))
    num_in = layers.Input(shape=(len(NUM_FEATURES),), dtype="float32", name="num")

    x = layers.Concatenate()(cat_vecs + [num_in])
    for i, units in enumerate(hidden):
        h = layers.Dense(units, kernel_regularizer=reg, name=f"dense_{i}")(x)
        h = layers.LayerNormalization(name=f"ln_{i}")(h)
        h = layers.Activation("swish")(h)
        h = layers.Dropout(dropout, name=f"drop_{i}")(h)
        skip = x if x.shape[-1] == units else layers.Dense(units, use_bias=False, name=f"skip_{i}")(x)
        x = layers.Add(name=f"res_{i}")([skip, h])
    deep = layers.Dense(1, name="deep_out")(x)
    wide = layers.Dense(1, name="wide_out")(num_in)
    out = layers.Add(name="listening_scaled")([deep, wide])

    model = keras.Model(cat_inputs + [num_in], out, name="podcast_wide_deep")
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay),
                  loss="mse", metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


def predict_scaled(model: keras.Model, x: dict, mc_dropout: bool = False, batch_size: int = 8192) -> np.ndarray:
    """Forward pass in batches. mc_dropout=True keeps Dropout active (Monte-Carlo dropout)."""
    n = len(x["num"])
    outs = []
    for s in range(0, n, batch_size):
        batch = {k: v[s:s + batch_size] for k, v in x.items()}
        outs.append(np.asarray(model(batch, training=mc_dropout)).ravel())
    return np.concatenate(outs)


def mc_dropout_predict(model, x, n_samples: int = 30) -> np.ndarray:
    """Returns array (n_samples, n_rows) of stochastic predictions (standardised)."""
    return np.stack([predict_scaled(model, x, mc_dropout=True) for _ in range(n_samples)])
