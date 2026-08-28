#!/usr/bin/env python3
"""Train and evaluate paper-aligned DeepGP-AFT simulation models."""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

EPS, Z975 = 1e-7, 1.959963984540054


class GlobalRawSigma(layers.Layer):
    """One residual scale shared by all subjects."""
    def build(self, input_shape):
        self.raw_sigma = self.add_weight(
            name="raw_sigma", shape=(),
            initializer=tf.keras.initializers.Constant(np.log(np.expm1(1.0))),
            trainable=True)

    def call(self, inputs):
        return tf.ones_like(inputs[..., :1]) * self.raw_sigma


def build_model(input_dim, widths, dropout, n_train):
    rho = 1.0 - dropout
    hidden_penalty = regularizers.L2(rho / (2.0 * n_train))
    output_penalty = regularizers.L2(1.0 / (2.0 * n_train))
    inputs = layers.Input(shape=(input_dim,), name="x")
    value = inputs
    for number, width in enumerate(widths, 1):
        value = layers.Dense(
            width, activation="relu", kernel_regularizer=hidden_penalty,
            bias_regularizer=hidden_penalty, name=f"hidden_{number}")(value)
        value = layers.Dropout(dropout, name=f"dropout_{number}")(value)
    mu = layers.Dense(1, activation="linear", kernel_regularizer=output_penalty,
                      bias_regularizer=output_penalty, name="mu")(value)
    raw_sigma = GlobalRawSigma(name="global_raw_sigma")(mu)
    return Model(inputs, layers.Concatenate()([mu, raw_sigma]), name="DeepGP_AFT")


def censored_lognormal_nll(y_true, y_pred):
    observed = tf.clip_by_value(y_true[:, 0], EPS, np.inf)
    event, mu = tf.cast(y_true[:, 1], y_pred.dtype), y_pred[:, 0]
    sigma = tf.nn.softplus(y_pred[:, 1]) + EPS
    log_y = tf.math.log(observed)
    z = (log_y - mu) / sigma
    log_pdf = (-log_y - tf.math.log(sigma)
               - 0.5 * tf.cast(np.log(2.0 * np.pi), y_pred.dtype)
               - 0.5 * tf.square(z))
    survival = 0.5 * tf.math.erfc(z / tf.cast(np.sqrt(2.0), y_pred.dtype))
    log_survival = tf.math.log(tf.clip_by_value(survival, EPS, 1.0))
    return tf.reduce_mean(-(event * log_pdf + (1.0 - event) * log_survival))


def feature_columns(frame):
    result, number = [], 1
    while f"x{number}" in frame.columns:
        result.append(f"x{number}")
        number += 1
    if not result:
        raise ValueError("No consecutive x1,...,xp columns found")
    return result


def load_split(path, columns):
    frame = pd.read_csv(path)
    missing = sorted((set(columns) | {"y", "delta", "logT"}) - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    x = frame[columns].to_numpy(np.float32)
    outcome = frame[["y", "delta"]].to_numpy(np.float32)
    if np.any(outcome[:, 0] <= 0) or not np.all(np.isin(outcome[:, 1], [0, 1])):
        raise ValueError(f"{path}: require y>0 and binary delta")
    return x, outcome, frame.logT.to_numpy(float)


def mc_draws(model, x, samples):
    return np.stack([model(x, training=True).numpy()[:, 0]
                     for _ in range(samples)], axis=0)


def event_residual_variance(observed, event, prediction):
    keep = event == 1
    if not keep.any():
        raise ValueError("No uncensored training observations")
    return float(np.mean((np.log(observed[keep]) - prediction[keep]) ** 2))


def censoring_survival_before(times, event):
    survival, before = 1.0, {}
    for value in np.unique(times):
        before[value] = survival
        survival *= 1.0 - np.sum((times == value) & (event == 0)) / np.sum(times >= value)
    return np.maximum(np.array([before[value] for value in times]), EPS)


def ipcw_c_index(times, event, prediction):
    times, event = np.asarray(times, float), np.asarray(event, int)
    g = censoring_survival_before(times, event)
    comparable = (times[:, None] < times[None, :]) & (event[:, None] == 1)
    weights = (event / g ** 2)[:, None] * comparable
    denominator = weights.sum()
    concordant = prediction[:, None] < prediction[None, :]
    return float((weights * concordant).sum() / denominator) if denominator else np.nan


def fit_seed(setting, seed, args):
    train_path, test_path = (setting / f"seed_{seed}_train.csv",
                             setting / f"seed_{seed}_test.csv")
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"missing train/test pair for seed {seed}")
    columns = feature_columns(pd.read_csv(train_path, nrows=1))
    x_train, y_train, _ = load_split(train_path, columns)
    x_test, y_test, true_log_t = load_split(test_path, columns)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.keras.backend.clear_session()
    model = build_model(len(columns), args.widths, args.dropout, len(x_train))
    model.compile(Adam(learning_rate=args.learning_rate), censored_lognormal_nll)
    started = time.perf_counter()
    history = model.fit(
        x_train, y_train, validation_split=args.validation_fraction, shuffle=True,
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=[EarlyStopping(monitor="val_loss", patience=args.patience,
                                 restore_best_weights=True)], verbose=args.verbose)
    train_draw, test_draw = (mc_draws(model, x_train, args.mc_samples),
                             mc_draws(model, x_test, args.mc_samples))
    train_mean, test_mean = train_draw.mean(0), test_draw.mean(0)
    residual_var = event_residual_variance(y_train[:, 0], y_train[:, 1], train_mean)
    total_sd = np.sqrt(residual_var + test_draw.var(0))
    lower, upper = test_mean - Z975 * total_sd, test_mean + Z975 * total_sd
    return {"seed": seed,
            "rmse": float(np.sqrt(np.mean((test_mean - true_log_t) ** 2))),
            "ipcw": ipcw_c_index(y_test[:, 0], y_test[:, 1], test_mean),
            "coverage": float(np.mean((true_log_t >= lower) & (true_log_t <= upper))),
            "interval": float(np.mean(upper - lower)),
            "epochs_trained": len(history.history["loss"]),
            "n_train": len(x_train), "n_test": len(x_test),
            "n_test_events": int(y_test[:, 1].sum()), "p": len(columns),
            "residual_variance": residual_var,
            "runtime_seconds": float(time.perf_counter() - started)}


def parse_widths(text):
    try:
        widths = tuple(int(item) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("widths must be comma-separated integers") from error
    if not widths or any(width <= 0 for width in widths):
        raise argparse.ArgumentTypeError("all widths must be positive")
    return widths


def run_setting(setting, args):
    output, rows = setting / "summary_updateAFT.csv", []
    for seed in range(args.first_seed, args.last_seed + 1):
        try:
            row = fit_seed(setting, seed, args)
        except FileNotFoundError:
            if args.require_all_seeds:
                raise
            print(f"[{setting.name}] seed={seed} skipped")
            continue
        rows.append(row)
        pd.DataFrame(rows).to_csv(output, index=False)
        print(f"[{setting.name}] {json.dumps(row, sort_keys=True)}")
    if not rows:
        raise FileNotFoundError(f"No usable train/test pairs in {setting}")
    print(f"saved {output}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--root", type=Path, help="run every n* child setting")
    location.add_argument("--dir", type=Path, help="run one setting")
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--last-seed", type=int, default=1099)
    parser.add_argument("--widths", type=parse_widths, default=(128, 128, 64))
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--mc-samples", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--require-all-seeds", action="store_true")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=0)
    args = parser.parse_args()
    settings = ([args.dir] if args.dir else
                sorted(path for path in args.root.iterdir()
                       if path.is_dir() and path.name.startswith("n")))
    if not settings:
        raise FileNotFoundError("No simulation setting directories found")
    for setting in settings:
        run_setting(setting, args)


if __name__ == "__main__":
    main()
