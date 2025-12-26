#!/usr/bin/env python3
"""
Train a lightweight learned transition model from NPZ data produced by
collect_transition_data.py.

Model: single linear layer → logits of shape (N_REG*N_REG)
Loss : cross-entropy vs. tiled next-belief target (matches runtime loader:
       np.tensordot(feats, W, axes=1) + b → reshape (5,5) → row softmax)

Saved artifact is compatible with HingePOMDP's learned_transition loader:
  np.savez(out_path, W=W, b=b, meta=meta)
"""

import argparse
import os
from typing import Tuple

import numpy as np

N_REG = 5


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(axis=1, keepdims=True), 1e-12, None)


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(float)
    y = data["y"].astype(float)
    meta = data.get("meta", {})
    return X, y, meta


def make_targets(y: np.ndarray) -> np.ndarray:
    """
    Expand next-belief targets (N,5) to a flattened (N,25) distribution by
    tiling the 5-D belief across rows. This aligns with the row-softmaxed
    5x5 transition matrix expected at runtime.
    """
    tiled = np.tile(y, (1, N_REG))
    tiled_sum = np.clip(tiled.sum(axis=1, keepdims=True), 1e-12, None)
    return tiled / tiled_sum


def train_linear(X: np.ndarray, y: np.ndarray, epochs: int, lr: float, l2: float, batch: int) -> Tuple[np.ndarray, np.ndarray]:
    n, f = X.shape
    k = N_REG * N_REG
    rng = np.random.default_rng(0)
    W = rng.normal(scale=0.01, size=(f, k))
    b = np.zeros(k, float)

    targets = make_targets(y)

    for ep in range(epochs):
        idx = rng.permutation(n)
        Xs = X[idx]; Ts = targets[idx]
        for i in range(0, n, batch):
            xb = Xs[i:i+batch]
            tb = Ts[i:i+batch]
            if xb.size == 0:
                continue
            logits = xb @ W + b
            probs = softmax(logits)
            grad_logits = (probs - tb) / xb.shape[0]
            grad_W = xb.T @ grad_logits + l2 * W
            grad_b = grad_logits.sum(axis=0)
            W -= lr * grad_W
            b -= lr * grad_b
    return W, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="NPZ produced by collect_transition_data.py")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--output", type=str, default="results/transition_training/transition_model.npz")
    args = ap.parse_args()

    X, y, meta_in = load_npz(args.npz)
    print(f"[load] X={X.shape}, y={y.shape}, meta={meta_in}")

    W, b = train_linear(X, y, epochs=args.epochs, lr=args.lr, l2=args.l2, batch=args.batch)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    meta_out = dict(
        history=meta_in.get("history", None),
        feature_dim=X.shape[1],
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        batch=args.batch,
        files=meta_in.get("files", None),
    )
    np.savez(args.output, W=W, b=b, meta=meta_out)
    print(f"[done] Saved model to {args.output} (W {W.shape}, b {b.shape})")


if __name__ == "__main__":
    main()
