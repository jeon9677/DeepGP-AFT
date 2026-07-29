#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepGP-like Log-normal AFT model with MC-Dropout + Uncertainty Quantification
==============================================================================

This module contains ONLY the model / training / evaluation / UQ code.
Data generation is assumed to live elsewhere; every function here takes
already-prepared arrays or DataFrames as input.

Expected data schema
--------------------
Each DataFrame (train / val / test) is expected to have:
    x1, x2, ..., xp   : covariates
    y                 : observed time = min(T, C, tau)
    delta             : event indicator (1 = event, 0 = censored)
    logT (optional)   : TRUE, uncensored log survival time. Only needed
                         for full-sample predictive-interval coverage
                         evaluation (an oracle quantity available in
                         simulations). If absent, coverage falls back
                         to event-only rows (where y == T exactly).

Module layout
-------------
  1. Model & loss           : build_deepgp_aft, aft_lognormal_nll
  2. Predictive UQ          : predict_mu_sigma_uq, prediction_interval,
                               coverage_at_alpha
  3. IPCW evaluation metrics: c_index_ipcw_rstyle, ipcw_rmse_logT
  4. Data-splitting utility : stratified_split
  5. High-level pipeline    : fit_model, evaluate_model,
                               train_and_evaluate  (glues everything
                               together for one seed)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as L, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PI = tf.constant(np.pi, dtype=tf.float32)
SQRT2 = tf.constant(np.sqrt(2.0), dtype=tf.float32)
LOG_HALF = tf.math.log(tf.constant(0.5, dtype=tf.float32))


def set_seed(seed: int = 1000) -> None:
    """Set numpy + tensorflow random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ===========================================================================
# 1. Model & loss
# ===========================================================================
def build_deepgp_aft(
    input_dim: int,
    width: int = 128,
    depth: int = 3,
    dropout: float = 0.2,
) -> Model:
    """
    Build the Deep-GP-like AFT network with MC-Dropout.

    Parameters
    ----------
    input_dim : number of covariates (p)
    width     : hidden layer width
    depth     : number of hidden (Dense + Dropout) blocks
    dropout   : dropout rate, kept active at inference time (MC-Dropout)

    Returns
    -------
    keras.Model that maps x -> [mu, raw_scale]
        mu(x)         : predicted mean of log(T)
        raw_scale(x)  : pre-softplus scale; sigma = softplus(raw_scale) + 1e-6

    Together (mu, sigma) fully specify the model's predictive
    distribution:  log(T) | x  ~  N(mu(x), sigma(x)^2)
    """
    inp = Input(shape=(input_dim,), name="x")
    x = inp
    for d in range(depth):
        x = L.Dense(width, activation="tanh", name=f"dense_{d}")(x)
        x = L.Dropout(dropout, name=f"drop_{d}")(x)
    out = L.Dense(2, activation=None, name="head_mu_raw")(x)
    return Model(inp, out, name="DeepGP_AFT")


def aft_lognormal_nll(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Log-normal AFT negative log-likelihood (right-censored).

    Parameters
    ----------
    y_true : tensor of shape (batch, 2) = [observed_time, event_indicator]
    y_pred : tensor of shape (batch, 2) = [mu, raw_scale]

    Returns
    -------
    scalar tensor: mean NLL over the batch
    """
    y = y_true[:, 0]
    delta = y_true[:, 1]
    mu = y_pred[:, 0]
    sigma = tf.nn.softplus(y_pred[:, 1]) + 1e-6

    y = tf.clip_by_value(y, 1e-30, 1e30)
    logy = tf.math.log(y)
    z = (logy - mu) / sigma

    logf = (
        -tf.math.log(y)
        - tf.math.log(sigma)
        - 0.5 * tf.math.log(2.0 * PI)
        - 0.5 * tf.square(z)
    )
    erfc_val = tf.clip_by_value(tf.math.erfc(z / SQRT2), 1e-45, 1.0)
    logS = LOG_HALF + tf.math.log(erfc_val)

    nll = -(delta * logf + (1.0 - delta) * logS)
    return tf.reduce_mean(nll)


# ===========================================================================
# 2. Predictive-distribution UQ (MC-Dropout)
# ===========================================================================
def predict_mu_sigma_uq(
    model: Model,
    X: np.ndarray,
    mc_passes: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte-Carlo Dropout inference with aleatoric/epistemic decomposition.

    The model defines log(T) | x ~ N(mu(x), sigma(x)^2). Running
    MC-dropout gives ``mc_passes`` samples of (mu, sigma) per input
    (a mixture of Gaussians), which is moment-matched to a single
    Gaussian (Kendall & Gal, 2017 style).

    Parameters
    ----------
    model     : trained keras.Model returned by build_deepgp_aft
    X         : (N, p) float array of covariates to predict on
    mc_passes : number of MC-Dropout forward passes

    Returns
    -------
    mu_mean          : (N,) predictive mean of log(T)
    sigma_aleatoric  : (N,) sqrt(E_m[sigma_m^2])   -- data-noise component
    sigma_epistemic  : (N,) sqrt(Var_m[mu_m])      -- dropout/model
                       uncertainty component
    sigma_total       : (N,) sqrt(aleatoric^2 + epistemic^2)
                        -- use this for prediction intervals
    """
    mu_samples = []
    sigma_samples = []
    for _ in range(mc_passes):
        raw = model(X, training=True).numpy()  # dropout stays active
        mu_samples.append(raw[:, 0])
        sigma_samples.append(np.log1p(np.exp(raw[:, 1])) + 1e-6)  # softplus
    mu_samples = np.stack(mu_samples, axis=0)       # (M, N)
    sigma_samples = np.stack(sigma_samples, axis=0)  # (M, N)

    mu_mean = mu_samples.mean(axis=0)
    sigma2_aleatoric = (sigma_samples ** 2).mean(axis=0)
    sigma2_epistemic = mu_samples.var(axis=0)
    sigma2_total = sigma2_aleatoric + sigma2_epistemic

    return (
        mu_mean,
        np.sqrt(sigma2_aleatoric),
        np.sqrt(sigma2_epistemic),
        np.sqrt(sigma2_total),
    )


def prediction_interval(
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    (1 - alpha) prediction interval for log(T) under a Gaussian
    predictive distribution.

    Parameters
    ----------
    mu_pred    : (N,) predictive mean (e.g. from predict_mu_sigma_uq)
    sigma_pred : (N,) predictive std to use (e.g. sigma_total)
    alpha      : miscoverage level (0.05 -> 95% interval)

    Returns
    -------
    lower, upper : (N,), (N,) interval bounds on the log(T) scale
    """
    z = norm.ppf(1 - alpha / 2)
    lower = mu_pred - z * sigma_pred
    upper = mu_pred + z * sigma_pred
    return lower, upper


def coverage_at_alpha(
    logT_true: np.ndarray,
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Empirical coverage of the (1 - alpha) prediction interval.

    Parameters
    ----------
    logT_true  : (N,) ground-truth log(T). May contain NaN for rows
                 without a known true value (e.g. censored rows when
                 no oracle logT is available) -- these are dropped.
    mu_pred    : (N,) predictive mean
    sigma_pred : (N,) predictive std (use sigma_total)
    alpha      : miscoverage level (0.05 -> 95% interval)

    Returns
    -------
    coverage  : fraction of (non-NaN) rows where logT_true falls inside
                [lower, upper]
    avg_width : average interval width
    """
    logT_true = np.asarray(logT_true, dtype=float)
    mu_pred = np.asarray(mu_pred, dtype=float)
    sigma_pred = np.asarray(sigma_pred, dtype=float)
    ok = np.isfinite(logT_true) & np.isfinite(mu_pred) & np.isfinite(sigma_pred)
    logT_true, mu_pred, sigma_pred = logT_true[ok], mu_pred[ok], sigma_pred[ok]
    if len(logT_true) == 0:
        return np.nan, np.nan

    lower, upper = prediction_interval(mu_pred, sigma_pred, alpha=alpha)
    inside = (logT_true >= lower) & (logT_true <= upper)
    return float(inside.mean()), float((upper - lower).mean())


# ===========================================================================
# 3. IPCW evaluation metrics
# ===========================================================================
def _km_survival_of_censoring(y: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Kaplan–Meier estimate of G(t) = P(C > t), evaluated at each y_i."""
    y = np.asarray(y, dtype=float)
    delta = np.asarray(delta, dtype=int)
    c = 1 - delta
    uniq = np.unique(y)
    r = np.array([(y >= t).sum() for t in uniq], dtype=float)
    d = np.array([c[y == t].sum() for t in uniq], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = 1.0 - np.divide(d, r, out=np.zeros_like(d), where=(r > 0))
    G_at_times = np.cumprod(step, dtype=float)
    idx = np.searchsorted(uniq, y, side="right") - 1
    G_hat = np.where(idx >= 0, G_at_times[idx], 1.0)
    return np.clip(G_hat, 1e-12, 1.0)


def c_index_ipcw_rstyle(
    y: np.ndarray,
    delta: np.ndarray,
    t_pred: np.ndarray,
    censor_prob: Optional[np.ndarray] = None,
) -> float:
    """
    IPCW C-index (Uno et al. style).

    Parameters
    ----------
    y           : (N,) observed times
    delta       : (N,) event indicators
    t_pred      : (N,) predicted risk score (e.g. predicted median time;
                  smaller = predicted to fail sooner)
    censor_prob : optional precomputed G_hat(y_i); if None, estimated
                  via Kaplan-Meier on the censoring distribution

    Returns
    -------
    IPCW C-index (float), or NaN if not enough comparable pairs
    """
    y = np.asarray(y, dtype=float)
    delta = np.asarray(delta, dtype=int)
    t_pred = np.asarray(t_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(delta) & np.isfinite(t_pred) & (y > 0)
    y, delta, t_pred = y[ok], delta[ok], t_pred[ok]
    n = len(y)
    if n < 2:
        return np.nan
    G_hat = (
        _km_survival_of_censoring(y, delta)
        if censor_prob is None
        else np.clip(np.asarray(censor_prob, dtype=float), 1e-12, 1.0)
    )
    num, den = 0.0, 0.0
    idx_all = np.arange(n)
    for i in range(n):
        if delta[i] != 1:
            continue
        w = delta[i] / (G_hat[i] ** 2)
        mask = (idx_all != i) & (y[i] < y)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        den += w * cnt
        num += w * int((mask & (t_pred[i] < t_pred)).sum())
    return float(num / den) if den > 0 else np.nan


def ipcw_rmse_logT(
    y: np.ndarray,
    delta: np.ndarray,
    mu_hat: np.ndarray,
    censor_prob: Optional[np.ndarray] = None,
) -> float:
    """
    IPCW-weighted RMSE of log(T):  w_i = delta_i / G_hat(y_i).

    Parameters
    ----------
    y, delta    : (N,) observed times / event indicators
    mu_hat      : (N,) predicted mean of log(T)
    censor_prob : optional precomputed G_hat(y_i)

    Returns
    -------
    IPCW-weighted RMSE (float), or NaN if no events
    """
    y = np.asarray(y, dtype=float)
    delta = np.asarray(delta, dtype=int)
    mu_hat = np.asarray(mu_hat, dtype=float)
    ok = np.isfinite(y) & np.isfinite(delta) & np.isfinite(mu_hat) & (y > 0)
    y, delta, mu_hat = y[ok], delta[ok], mu_hat[ok]
    G_hat = (
        _km_survival_of_censoring(y, delta)
        if censor_prob is None
        else np.clip(np.asarray(censor_prob, dtype=float), 1e-12, 1.0)
    )
    w = delta / G_hat
    num = np.sum(w * (np.log(y) - mu_hat) ** 2)
    den = np.sum(w)
    return float(np.sqrt(num / den)) if den > 0 else np.nan


# ===========================================================================
# 4. Data-splitting utility (model-side train/val split, NOT data generation)
# ===========================================================================
def stratified_split(
    df: pd.DataFrame,
    label_col: str = "delta",
    frac: float = 0.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into (A, B), stratified by a binary label column,
    with A receiving ``frac`` proportion of each class.

    Used e.g. to split a 70%-train_pool into a 5/7 train_core and a
    2/7 val split (giving overall 50%/20%/30% train/val/test).

    Parameters
    ----------
    df        : input DataFrame (must contain label_col)
    label_col : name of the binary stratification column
    frac      : proportion assigned to A, per class
    seed      : random seed for reproducibility

    Returns
    -------
    A, B : two DataFrames (shuffled, index reset)
    """
    rng = np.random.default_rng(seed)
    A_parts, B_parts = [], []
    for val in sorted(df[label_col].unique()):
        block = df[df[label_col] == val]
        idx = block.index.to_numpy()
        rng.shuffle(idx)
        n_A = int(np.floor(frac * len(idx)))
        A_parts.append(block.loc[idx[:n_A]])
        B_parts.append(block.loc[idx[n_A:]])
    A = pd.concat(A_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    B = pd.concat(B_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return A, B


def df_to_xy(df: pd.DataFrame, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (X, y_target) arrays from a DataFrame.

    Parameters
    ----------
    df : DataFrame with columns x1..xp, y, delta
    p  : number of covariates

    Returns
    -------
    X        : (N, p) float32 covariates
    y_target : (N, 2) float32, columns = [observed_time, event_indicator]
    """
    X = df[[f"x{j + 1}" for j in range(p)]].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)
    d = df["delta"].to_numpy(dtype=np.float32)
    y_target = np.stack([y, d], axis=1).astype(np.float32)
    return X, y_target


# ===========================================================================
# 5. High-level pipeline: fit + evaluate for one train/val/test triple
# ===========================================================================
def fit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    p: int,
    width: int = 128,
    depth: int = 3,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 128,
    epochs: int = 500,
    patience: int = 20,
    seed: Optional[int] = None,
) -> Model:
    """
    Build and train a DeepGP-AFT model.

    Parameters
    ----------
    X_train, y_train : training arrays (y_train columns = [time, delta])
    X_val, y_val      : validation arrays, used for early stopping / LR decay
    p                 : number of covariates
    width, depth, dropout, lr, batch_size, epochs, patience : hyperparameters
    seed              : optional random seed (sets numpy + tf seeds)

    Returns
    -------
    trained keras.Model
    """
    if seed is not None:
        set_seed(seed)
    tf.keras.backend.clear_session()

    model = build_deepgp_aft(input_dim=p, width=width, depth=depth, dropout=dropout)
    model.compile(optimizer=Adam(lr), loss=aft_lognormal_nll)

    callbacks = [
        EarlyStopping(patience=patience, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model


def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    logT_true: Optional[np.ndarray] = None,
    mc_passes: int = 30,
    alpha: float = 0.05,
) -> dict:
    """
    Evaluate a trained model on a test set: point metrics + predictive
    UQ (prediction interval + coverage).

    Parameters
    ----------
    model     : trained keras.Model
    X_test    : (N, p) test covariates
    y_test    : (N, 2) test targets, columns = [observed_time, event_indicator]
    logT_true : optional (N,) oracle true log(T), used for full-sample
                coverage. If None, coverage falls back to event-only
                rows (y == T there).
    mc_passes : number of MC-Dropout passes
    alpha     : prediction-interval miscoverage level (0.05 -> 95%)

    Returns
    -------
    dict with keys:
        n_test, n_events,
        rmse_logT, mae_logT, rmse_time_median, mae_time_median   (event-only),
        ipcw_rmse_logT, cindex_median_ipcw,
        coverage_95, avg_pi_width_95,
        mean_sigma_aleatoric, mean_sigma_epistemic, mean_sigma_total
    """
    y_time = y_test[:, 0]
    d_test = y_test[:, 1]

    mu_hat, sigma_aleatoric, sigma_epistemic, sigma_total = predict_mu_sigma_uq(
        model, X_test, mc_passes=mc_passes
    )
    medT_hat = np.exp(mu_hat)

    def _rmse(a, b):
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    def _mae(a, b):
        return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

    evt = d_test == 1
    if evt.sum() > 0:
        rmse_logT = _rmse(np.log(y_time[evt]), mu_hat[evt])
        mae_logT = _mae(np.log(y_time[evt]), mu_hat[evt])
        rmse_time_med = _rmse(y_time[evt], medT_hat[evt])
        mae_time_med = _mae(y_time[evt], medT_hat[evt])
    else:
        rmse_logT = mae_logT = rmse_time_med = mae_time_med = np.nan

    ipcw_rmse = ipcw_rmse_logT(y_time, d_test, mu_hat)
    cindex_ipcw = c_index_ipcw_rstyle(y_time, d_test, medT_hat)

    if logT_true is None:
        logT_true = np.full_like(mu_hat, np.nan, dtype=np.float32)
        logT_true[evt] = np.log(y_time[evt])

    coverage_95, avg_pi_width_95 = coverage_at_alpha(
        logT_true, mu_hat, sigma_total, alpha=alpha
    )

    return dict(
        n_test=len(y_time),
        n_events=int(evt.sum()),
        rmse_logT=rmse_logT,
        mae_logT=mae_logT,
        rmse_time_median=rmse_time_med,
        mae_time_median=mae_time_med,
        ipcw_rmse_logT=ipcw_rmse,
        cindex_median_ipcw=cindex_ipcw,
        coverage_95=coverage_95,
        avg_pi_width_95=avg_pi_width_95,
        mean_sigma_aleatoric=float(np.nanmean(sigma_aleatoric)),
        mean_sigma_epistemic=float(np.nanmean(sigma_epistemic)),
        mean_sigma_total=float(np.nanmean(sigma_total)),
    )


def train_and_evaluate(
    train_pool_df: pd.DataFrame,
    test_df: pd.DataFrame,
    p: int,
    seed: int = 1000,
    val_frac_within_train: float = 2.0 / 7.0,
    width: int = 128,
    depth: int = 3,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 128,
    epochs: int = 500,
    patience: int = 20,
    mc_passes: int = 30,
    alpha: float = 0.05,
) -> dict:
    """
    One-call convenience wrapper: split -> fit -> evaluate, for a single
    seed / single dataset.

    Parameters
    ----------
    train_pool_df : DataFrame with the full (70%) train pool, columns
                    x1..xp, y, delta (and optionally logT)
    test_df       : DataFrame with the held-out (30%) test set, same
                    columns (and optionally logT, used for coverage)
    p             : number of covariates
    seed          : random seed (controls both the train/val split and
                    model initialization)
    val_frac_within_train : fraction of train_pool assigned to val
                    (default 2/7, so overall split is 50/20/30)
    width, depth, dropout, lr, batch_size, epochs, patience : model
                    hyperparameters
    mc_passes     : MC-Dropout passes at inference
    alpha         : prediction-interval level

    Returns
    -------
    dict of metrics (see evaluate_model), plus:
        seed            : the seed used
        runtime_seconds : wall-clock training time
    """
    val_df, train_core_df = stratified_split(
        train_pool_df, label_col="delta", frac=val_frac_within_train, seed=seed
    )

    X_tr, y_tr = df_to_xy(train_core_df, p)
    X_va, y_va = df_to_xy(val_df, p)
    X_te, y_te = df_to_xy(test_df, p)

    t0 = time.time()
    model = fit_model(
        X_tr, y_tr, X_va, y_va, p=p,
        width=width, depth=depth, dropout=dropout,
        lr=lr, batch_size=batch_size, epochs=epochs, patience=patience,
        seed=seed,
    )
    runtime = time.time() - t0

    logT_true = (
        test_df["logT"].to_numpy(dtype=np.float32)
        if "logT" in test_df.columns
        else None
    )

    metrics = evaluate_model(
        model, X_te, y_te, logT_true=logT_true, mc_passes=mc_passes, alpha=alpha
    )
    metrics["seed"] = seed
    metrics["runtime_seconds"] = runtime
    return metrics
