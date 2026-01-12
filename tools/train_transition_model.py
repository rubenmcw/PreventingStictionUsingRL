#!/usr/bin/env python3
"""
Train a lightweight learned transition model from NPZ data produced by
collect_transition_data.py.

Model: multi-layer perceptron → logits of shape (N_REG*N_REG)
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
    meta_raw = data["meta"] if "meta" in data.files else {}
    # meta may be stored as a 0-d array, a length-1 object array, or already a dict
    if isinstance(meta_raw, np.ndarray):
        try:
            if meta_raw.shape == () or meta_raw.size == 1:
                meta_raw = meta_raw.item()
        except Exception:
            pass
    if not isinstance(meta_raw, dict):
        meta_raw = {}
    return X, y, meta_raw


def make_targets(y: np.ndarray) -> np.ndarray:
    """
    Expand next-belief targets (N,5) to a flattened (N,25) distribution by
    tiling the 5-D belief across rows. This aligns with the row-softmaxed
    5x5 transition matrix expected at runtime.
    """
    tiled = np.tile(y, (1, N_REG))
    tiled_sum = np.clip(tiled.sum(axis=1, keepdims=True), 1e-12, None)
    return tiled / tiled_sum


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def train_mlp(X: np.ndarray, y: np.ndarray, hidden: Tuple[int, ...], epochs: int, lr: float, l2: float, batch: int) -> Tuple[list, list]:
    n, f = X.shape
    k = N_REG * N_REG
    rng = np.random.default_rng(0)
    layer_sizes = (f,) + hidden + (k,)
    weights = []
    biases = []
    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        weights.append(rng.normal(scale=0.01, size=(in_dim, out_dim)))
        biases.append(np.zeros(out_dim, float))

    targets = make_targets(y)

    for _ in range(epochs):
        idx = rng.permutation(n)
        Xs = X[idx]; Ts = targets[idx]
        for i in range(0, n, batch):
            xb = Xs[i:i+batch]
            tb = Ts[i:i+batch]
            if xb.size == 0:
                continue
            activations = [xb]
            preacts = []
            for layer_idx, (W, b) in enumerate(zip(weights, biases)):
                z = activations[-1] @ W + b
                preacts.append(z)
                if layer_idx < len(weights) - 1:
                    activations.append(relu(z))
                else:
                    activations.append(z)
            logits = activations[-1]
            probs = softmax(logits)
            grad = (probs - tb) / xb.shape[0]
            for layer_idx in range(len(weights) - 1, -1, -1):
                W = weights[layer_idx]
                a_prev = activations[layer_idx]
                grad_W = a_prev.T @ grad + l2 * W
                grad_b = grad.sum(axis=0)
                weights[layer_idx] = W - lr * grad_W
                biases[layer_idx] -= lr * grad_b
                if layer_idx > 0:
                    grad = grad @ W.T
                    grad = grad * (preacts[layer_idx - 1] > 0)
    return weights, biases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="NPZ produced by collect_transition_data.py")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=str, default="64,64",
                    help="Comma-separated hidden layer sizes (e.g. '64,32'). Use empty string for linear.")
    ap.add_argument("--output", type=str, default="results/transition_training/transition_model.npz")
    args = ap.parse_args()

    X, y, meta_in = load_npz(args.npz)
    print(f"[load] X={X.shape}, y={y.shape}, meta={meta_in}")

    hidden = tuple(int(h) for h in args.hidden.split(",") if h.strip())
    weights, biases = train_mlp(X, y, hidden=hidden, epochs=args.epochs, lr=args.lr, l2=args.l2, batch=args.batch)

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
        hidden=hidden,
        files=meta_in.get("files", None),
    )
    savez_payload = {"meta": meta_out}
    for idx, (W, b) in enumerate(zip(weights, biases)):
        savez_payload[f"W{idx}"] = W
        savez_payload[f"b{idx}"] = b
    if len(weights) == 1:
        savez_payload["W"] = weights[0]
        savez_payload["b"] = biases[0]
    np.savez(args.output, **savez_payload)
    last_shape = (weights[-1].shape, biases[-1].shape)
    print(f"[done] Saved model to {args.output} (layers {len(weights)}, last {last_shape})")


if __name__ == "__main__":
    main()
