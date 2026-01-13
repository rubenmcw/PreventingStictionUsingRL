from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class DeltaUOptimizerConfig:
    horizon: int
    tau_sat: float
    du_max: float
    cem_samples: int
    cem_elites: int
    cem_iters: int
    cem_init_std: float
    mppi_samples: int
    mppi_lambda: float
    mppi_sigma: float
    noise_smooth_window: int = 3


def smooth_noise(noise: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return noise
    kernel = np.ones(window, dtype=float)
    kernel /= kernel.sum()
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 1, noise)


def integrate_delta_u(delta_u_seq: np.ndarray, u_prev: float, du_max: float, tau_sat: float) -> np.ndarray:
    u_seq = np.empty_like(delta_u_seq, dtype=float)
    delta0 = np.clip(delta_u_seq[0], -du_max, du_max)
    u_seq[0] = np.clip(u_prev + delta0, -tau_sat, tau_sat)
    for t in range(1, len(delta_u_seq)):
        delta = np.clip(delta_u_seq[t], -du_max, du_max)
        u_seq[t] = np.clip(u_seq[t - 1] + delta, -tau_sat, tau_sat)
    return u_seq


class DeltaUSequenceOptimizer:
    def __init__(self, config: DeltaUOptimizerConfig, rng: Optional[np.random.Generator] = None):
        self.config = config
        self.rng = rng or np.random.default_rng()
        self.last_u_seq_best: Optional[np.ndarray] = None

    def _warm_start_u_nom(self) -> np.ndarray:
        H = self.config.horizon
        if self.last_u_seq_best is None:
            return np.zeros(H, dtype=float)
        u_nom = np.empty(H, dtype=float)
        u_nom[:-1] = self.last_u_seq_best[1:]
        u_nom[-1] = self.last_u_seq_best[-1]
        return u_nom

    @staticmethod
    def _delta_from_u_nom(u_nom: np.ndarray, u_prev: float) -> np.ndarray:
        delta = np.empty_like(u_nom, dtype=float)
        delta[0] = u_nom[0] - u_prev
        delta[1:] = np.diff(u_nom)
        return delta

    def cem_optimize(self, cost_fn: Callable[[np.ndarray], float], u_prev: float) -> Tuple[float, np.ndarray]:
        cfg = self.config
        H = cfg.horizon
        u_nom = self._warm_start_u_nom()
        mean = self._delta_from_u_nom(u_nom, u_prev)
        std_init = max(1e-6, min(cfg.cem_init_std, cfg.du_max))
        std = np.full(H, std_init, dtype=float)
        n_samples = int(cfg.cem_samples)
        n_elites = int(cfg.cem_elites)
        n_iters = int(cfg.cem_iters)
        best_u_seq = u_nom.copy()
        best_c = np.inf

        for _ in range(n_iters):
            noise = self.rng.standard_normal((n_samples, H))
            noise = smooth_noise(noise, cfg.noise_smooth_window)
            delta_samples = mean[None, :] + std[None, :] * noise
            delta_samples = np.clip(delta_samples, -cfg.du_max, cfg.du_max)
            costs = np.empty(n_samples, dtype=float)
            for i in range(n_samples):
                u_seq = integrate_delta_u(delta_samples[i], u_prev, cfg.du_max, cfg.tau_sat)
                costs[i] = cost_fn(u_seq)
            elite_idx = np.argsort(costs)[:n_elites]
            elites = delta_samples[elite_idx]
            mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-6
            if costs[elite_idx[0]] < best_c:
                best_c = float(costs[elite_idx[0]])
                best_u_seq = integrate_delta_u(elites[0], u_prev, cfg.du_max, cfg.tau_sat)

        self.last_u_seq_best = best_u_seq
        return float(best_u_seq[0]), best_u_seq

    def mppi_optimize(self, cost_fn: Callable[[np.ndarray], float], u_prev: float) -> Tuple[float, np.ndarray]:
        cfg = self.config
        H = cfg.horizon
        u_nom = self._warm_start_u_nom()
        delta_nom = self._delta_from_u_nom(u_nom, u_prev)

        noise = self.rng.standard_normal((cfg.mppi_samples, H))
        noise = smooth_noise(noise, cfg.noise_smooth_window)
        noise_scaled = noise * cfg.mppi_sigma
        delta_samples = delta_nom[None, :] + noise_scaled
        delta_samples = np.clip(delta_samples, -cfg.du_max, cfg.du_max)

        costs = np.empty(cfg.mppi_samples, dtype=float)
        for i in range(cfg.mppi_samples):
            u_seq = integrate_delta_u(delta_samples[i], u_prev, cfg.du_max, cfg.tau_sat)
            costs[i] = cost_fn(u_seq)

        costs -= float(costs.min())
        lam = max(cfg.mppi_lambda, 1e-6)
        scaled = np.clip(costs / lam, 0.0, 50.0)
        weights = np.exp(-scaled)
        weights_sum = weights.sum()
        if weights_sum > 0.0:
            delta_nom = delta_nom + (weights[:, None] * noise_scaled).sum(axis=0) / weights_sum
        delta_nom = np.clip(delta_nom, -cfg.du_max, cfg.du_max)
        u_seq_nom = integrate_delta_u(delta_nom, u_prev, cfg.du_max, cfg.tau_sat)
        self.last_u_seq_best = u_seq_nom
        return float(u_seq_nom[0]), u_seq_nom
