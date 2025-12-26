#!/usr/bin/env python3
"""
Collect transition-training data from hinge POMDP telemetry CSVs.

Inputs (per row):
  - tau_cmd (u), dq, angle (theta)
  - belief components b0..b4 (target distribution at next step)

Outputs:
  - NPZ containing:
      X  : features matching POMDP._transition_features
      y  : next-step belief distribution (5,)
      meta: dict with history length and source files
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

N_REG = 5


def _build_features(u: np.ndarray, dq: np.ndarray, theta: np.ndarray, idx: int, history: int) -> np.ndarray:
    base = [u[idx], dq[idx], theta[idx], abs(u[idx]), abs(dq[idx]), 1.0]
    if history <= 0:
        return np.asarray(base, float)
    start = idx - history
    hist_block = np.stack([u[start:idx], dq[start:idx], theta[start:idx]], axis=1).reshape(-1)
    return np.concatenate([np.asarray(base, float), hist_block])


def collect_from_csv(path: str, history: int) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    required = ["tau_cmd", "dq", "angle", "b0", "b1", "b2", "b3", "b4"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    u = df["tau_cmd"].to_numpy(float)
    dq = df["dq"].to_numpy(float)
    th = df["angle"].to_numpy(float)
    beliefs = df[["b0", "b1", "b2", "b3", "b4"]].to_numpy(float)

    feats: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    # need k -> k+1 target, so stop at len-1
    for i in range(history, len(df) - 1):
        feats.append(_build_features(u, dq, th, i, history))
        targets.append(beliefs[i + 1])

    if not feats:
        return np.empty((0, 0)), np.empty((0, N_REG))
    X = np.vstack(feats)
    y = np.vstack(targets)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="CSV files to aggregate (e.g., POMDP-*.csv)")
    ap.add_argument("--history", type=int, default=6, help="history length for (u,dq,theta) tuples")
    ap.add_argument("--output", type=str, default=os.path.join("results", "transition_training", "transition_data.npz"),
                    help="output NPZ path")
    args = ap.parse_args()

    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    for p in args.inputs:
        X, y = collect_from_csv(p, args.history)
        if X.size == 0:
            continue
        X_all.append(X)
        y_all.append(y)
        print(f"[ok] {p}: {len(X)} samples")

    if not X_all:
        raise RuntimeError("No samples collected; check input files or history length.")

    X = np.vstack(X_all)
    y = np.vstack(y_all)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    meta = dict(history=args.history, files=args.inputs, feature_dim=X.shape[1])
    np.savez(args.output, X=X, y=y, meta=meta)
    print(f"[done] Saved {len(X)} samples to {args.output} (feature_dim={X.shape[1]})")


if __name__ == "__main__":
    main()
