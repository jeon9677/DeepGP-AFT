#!/usr/bin/env python3
"""Generate the simulation data described in the DeepGP-AFT manuscript."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def ar1_covariance(p: int, rho: float = 0.15) -> np.ndarray:
    index = np.arange(p)
    return rho ** np.abs(index[:, None] - index[None, :])


def mean_function(x: np.ndarray) -> np.ndarray:
    if x.shape[1] < 5:
        raise ValueError("The manuscript simulation requires p >= 5")
    value = (x[:, 0] * x[:, 1] + 0.5 * x[:, 2] ** 2
             + np.sin(np.pi * x[:, 3] / 2.0) - 0.8 * x[:, 4])
    if x.shape[1] > 5:
        weights = np.linspace(0.2, -0.1, x.shape[1] - 5)
        value += (x[:, 5:] ** 3) @ weights
    return value


def generate_dataset(n, p, error_variance, tau, rho, seed):
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(np.zeros(p), ar1_covariance(p, rho), size=n)
    mu = mean_function(x)
    log_t = mu + rng.normal(0.0, np.sqrt(error_variance), size=n)
    event_time = np.exp(log_t)
    censor_time = rng.uniform(0.0, tau, size=n)
    frame = pd.DataFrame(x, columns=[f"x{j}" for j in range(1, p + 1)])
    frame["y"] = np.minimum(event_time, censor_time)
    frame["delta"] = (event_time <= censor_time).astype(np.int8)
    frame["logT"] = log_t
    frame["mu_true"] = mu
    frame["sigma_true"] = np.sqrt(error_variance)
    return frame


def split_dataset(frame, train_fraction, seed):
    rng = np.random.default_rng(seed + 10_000_000)
    order = rng.permutation(len(frame))
    cut = int(np.floor(train_fraction * len(frame)))
    return (frame.iloc[order[:cut]].reset_index(drop=True),
            frame.iloc[order[cut:]].reset_index(drop=True))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=Path("simul_AFT/SimulData"))
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--p", type=int, choices=[5, 30, 100], required=True)
    parser.add_argument("--error-variance", type=float, choices=[0.1, 0.25], required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--rho", type=float, default=0.15)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=1000)
    args = parser.parse_args()
    if args.tau <= 0 or not 0 < args.train_fraction < 1:
        parser.error("tau must be positive and train-fraction must be in (0,1)")

    setting = f"n{args.n}_p{args.p}_sigma{args.error_variance}_tau{args.tau}"
    destination = args.outdir / setting
    destination.mkdir(parents=True, exist_ok=True)
    metadata = []
    for seed in range(args.first_seed, args.first_seed + args.replicates):
        frame = generate_dataset(
            args.n, args.p, args.error_variance, args.tau, args.rho, seed)
        train, test = split_dataset(frame, args.train_fraction, seed)
        train.to_csv(destination / f"seed_{seed}_train.csv", index=False)
        test.to_csv(destination / f"seed_{seed}_test.csv", index=False)
        metadata.append({"seed": seed, "n_train": len(train), "n_test": len(test),
                         "censoring_all": 1.0 - frame.delta.mean(),
                         "censoring_train": 1.0 - train.delta.mean(),
                         "censoring_test": 1.0 - test.delta.mean()})
        print(f"[{setting}] seed={seed} saved")
    pd.DataFrame(metadata).to_csv(destination / "simulation_metadata.csv", index=False)


if __name__ == "__main__":
    main()
