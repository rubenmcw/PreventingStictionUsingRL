#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage A — Python-only hinge simulations (PID / MPC / POMDP)
-----------------------------------------------------------
• 1-DoF hinge:  J*θ̈ + D*θ̇ + τ_well(θ) + τ_fric(θ̇) = u + τ_load
• Multi-well potentials: Light / Nominal / Severe
• Smoothed Coulomb/Stribeck friction
• Controllers: PID, MPC (constant-input short horizon), POMDP (belief over friction scale)
• CSV per (controller, profile) + summary CSV/TeX + figures (time–work tradeoff, traces)

Outputs:
  ./results/python_sim/time_work_tradeoff_python.pdf
  ./results/python_sim/state_traces_python.pdf
  ./stageA_summary.csv
  ./stageA_summary.tex
  ./<CTRL>-<PROFILE>.csv   (9 files)

No MuJoCo dependency.
"""

from __future__ import annotations
import os, sys, math, argparse, itertools, random
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------- Config (edit here) -----------------------------

DT           = 1/200.0          # control + integration timestep (s)
T_TOTAL      = 10.0             # total horizon (s)
TAU_SAT      = 5.0              # |u| <= TAU_SAT (N·m)
SP_STEP_DEG  = 40.0             # target step (deg)
SP_SPEED     = math.radians(12) # setpoint slew (rad/s)
J_EQ         = 0.08             # kg·m^2
D_EQ         = 0.8              # N·m·s/rad viscous damping
NOISE_STD    = 0.0              # process noise on θ̈ (rad/s^2), scaled by sqrt(DT)

# friction (smoothed Coulomb / Stribeck)
FS = 0.60 * TAU_SAT    # static
FC = 0.35 * TAU_SAT    # kinetic
VS = 0.05              # Stribeck speed (rad/s)
V_EPS = 1e-3           # for tanh smoothing

# multi-well potential (τ_well = K * sin(n θ))
# profiles per Stage A: Light / Nominal / Severe (increasing K)
WELL_N = 6  # number of wells over 2π
PROFILES = {
    "Light":   dict(K=0.60),
    "Nominal": dict(K=1.40),
    "Severe":  dict(K=2.50),
}

# PID gains
PID_KP, PID_KI, PID_KD   = 12.0, 2.0, 1.0
PID_I_FRAC               = 0.35   # I clamp as fraction of TAU_SAT

# MPC horizon / weights
MPC_H, MPC_GAMMA         = 12, 0.98
W_TH, W_W, W_U, W_DU     = 50.0, 0.5, 8e-5, 2e-4
W_ERR_SOFT               = 0.02

# POMDP: action grid, belief over friction scale {0.85, 1.00, 1.15}×(FS,FC)
N_ACTIONS = 257
_z = np.linspace(-1.0, 1.0, N_ACTIONS)
ACTIONS = TAU_SAT * np.sign(_z) * (np.abs(_z)**1.7)
FRIC_SCALES = np.array([0.85, 1.00, 1.15])  # uncertainty over true Stribeck levels
SIGMA_RESIDUAL = 0.30                        # N·m (likelihood width)
P_MAX_RISK = 0.4
TAU_MAX_FOR_RISK = 1.3 * TAU_SAT
ERR_RISK_THRESH = math.radians(4.6)
W_RISK_ACT, W_RISK_ERR = 1.5, 1.0

# metrics
ESC_TOL_DEG = 2.0
ESC_DWELL_S = 0.50
SAT_FRAC    = 0.98

OUT_DIR = os.path.join("results", "python_sim")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------- Plant model ---------------------------------

def tau_fric_smooth(omega: float, Fs: float, Fc: float) -> float:
    """Stribeck-like friction with smooth sign; torque opposes motion."""
    s = abs(omega)
    level = Fc + (Fs - Fc) * math.exp(- (s / VS)**2)
    return level * math.tanh(omega / V_EPS)

def tau_well(theta: float, K: float, n: int = WELL_N) -> float:
    """Detent torque from multi-well potential."""
    return K * math.sin(n * theta)

@dataclass
class HingeState:
    theta: float
    omega: float

@dataclass
class PlantParams:
    J: float = J_EQ
    D: float = D_EQ
    Fs: float = FS
    Fc: float = FC
    K: float = PROFILES["Nominal"]["K"]
    n: int = WELL_N

class HingePlant:
    def __init__(self, params: PlantParams, noise_std: float = NOISE_STD):
        self.p = params
        self.noise_std = float(noise_std)

    def step(self, x: HingeState, u: float, dt: float) -> HingeState:
        # Dynamics: J ẇ = u - D w - τ_fric(w) - τ_well(θ)
        tf = tau_fric_smooth(x.omega, self.p.Fs, self.p.Fc)
        tw = tau_well(x.theta, self.p.K, self.p.n)
        a = (u - self.p.D * x.omega - tf - tw) / self.p.J
        if self.noise_std > 0.0:
            a += self.noise_std * np.random.randn() / math.sqrt(max(dt, 1e-9))
        w = x.omega + dt * a
        th = x.theta + dt * w
        return HingeState(th, w), dict(tau_fric=tf, tau_well=tw, acc=a)

# ------------------------------ Setpoint ramp --------------------------------

class Setpoint:
    def __init__(self, q0: float, goal: float, speed: float):
        self.sp = float(q0)
        self.goal = float(goal)
        self.speed = float(speed)

    def step(self, dt: float) -> Tuple[float, float]:
        err = self.goal - self.sp
        max_step = self.speed * dt
        if abs(err) <= max_step:
            self.sp = self.goal
            sp_dot = 0.0
        else:
            self.sp += math.copysign(max_step, err)
            sp_dot = math.copysign(self.speed, err)
        return self.sp, sp_dot

# -------------------------------- Controllers --------------------------------

class PIDCtrl:
    def __init__(self, kp, ki, kd, tau_sat=TAU_SAT, i_frac=PID_I_FRAC, sp_speed=SP_SPEED):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.tau_sat = float(tau_sat)
        self.i_lim = float(i_frac) * self.tau_sat
        self.I = 0.0
        self._sp = 0.0
        self._goal = 0.0
        self._sp_prev = 0.0
        self._sp_speed = float(sp_speed)
        self.last_u = 0.0

    def reset(self, q0: float):
        self._sp = self._goal = float(q0)
        self._sp_prev = self._sp
        self.I = 0.0
        self.last_u = 0.0

    def set_goal(self, q_goal: float):
        self._goal = float(q_goal)

    def step(self, x: HingeState, dt: float) -> Tuple[float, float, float]:
        # setpoint ramp
        sp, sp_dot = Setpoint(self._sp, self._goal, self._sp_speed).step(dt)
        self._sp = sp
        e = sp - x.theta
        e_dot = sp_dot - x.omega

        # anti-windup: integrate unless saturating in the same direction
        u_raw = self.kp*e + self.kd*e_dot + self.ki*self.I
        will_sat = abs(u_raw) > self.tau_sat + 1e-12
        if not will_sat or (u_raw*e) < 0.0:
            self.I += e * dt
            if self.ki > 0:
                self.I = max(-self.i_lim/self.ki, min(self.I, self.i_lim/self.ki))
        u = self.kp*e + self.kd*e_dot + self.ki*self.I
        u = max(-self.tau_sat, min(self.tau_sat, u))
        self.last_u = u
        return u, sp, self._goal

class MPCCtrl:
    """Constant-input enumeration MPC with friction-aware prediction,
    breakaway encouragement, and a brief 'kick' when stuck.
    Public interface unchanged from the original."""
    def __init__(self, J=J_EQ, D=D_EQ, H=MPC_H, gamma=MPC_GAMMA,
                 w_th=W_TH, w_w=W_W, w_u=W_U, w_du=W_DU, w_l1=W_ERR_SOFT,
                 tau_sat=TAU_SAT, sp_speed=SP_SPEED):
        # Plant model (linear) + cost weights
        self.J, self.D = float(J), float(D)
        self.H = int(max(30, H))             # ensure a minimally useful horizon
        self.gamma = float(gamma)
        self.w_th, self.w_w = float(w_th), float(w_w)
        # make torque cost gentle so MPC will actually push through stiction
        self.w_u = float(max(2e-5, w_u))
        self.w_du, self.w_l1 = float(w_du), float(w_l1)

        # Limits / setpoint ramp
        self.tau_sat = float(tau_sat)
        self._sp = 0.0
        self._goal = 0.0
        self._sp_speed = float(sp_speed)

        # Last command
        self.last_u = 0.0

        # --- Additions for robustness ---
        # small integral bias to cancel steady-state offsets
        self.bias_u = 0.0
        self.Ki_bias = 0.6
        self.bias_limit = 0.35 * self.tau_sat

        # friction/disturbance estimate τ̂_f from torque residuals
        self.tau_f_est = 0.0
        self.ff_alpha = 0.25
        self.ff_clip = 0.70 * self.tau_sat

        # filtered acceleration estimate
        self._a_hat = 0.0
        self._prev_v = 0.0

        # break-away & kick logic
        self.v_small = 0.03          # rad/s : "stuck" speed
        self.e_break = 0.06          # rad   : "nontrivial error"
        self.r_break = 0.15          # penalty weight when |u| below breakaway
        self.stuck_t = 0.0
        self.stuck_thresh_s = 0.40
        self.kick_dur_s = 0.12
        self._kick_left = 0.0

    def reset(self, q0: float):
        self._sp = self._goal = float(q0)
        self.last_u = 0.0
        self.bias_u = 0.0
        self.tau_f_est = 0.0
        self._a_hat = 0.0
        self._prev_v = 0.0
        self.stuck_t = 0.0
        self._kick_left = 0.0

    def set_goal(self, q_goal: float):
        self._goal = float(q_goal)

    def _rollout_cost_const_u(self, th0: float, w0: float, u: float,
                              sp_hold: float, u_prev: float,
                              tau_f_est: float, dt: float) -> float:
        """Discounted cost over the horizon with constant u and constant τ̂_f."""
        th, w = th0, w0
        csum = self.w_du * (u - u_prev)**2
        for h in range(self.H):
            e = (th - sp_hold)
            w_disc = (self.gamma**h)
            csum += w_disc * (self.w_th*(e*e) + self.w_w*(w*w) + self.w_u*(u*u) + self.w_l1*abs(e))
            # linear model + disturbance/friction offset
            w = w + dt * ((u - tau_f_est - self.D*w) / self.J)
            th = th + dt * w
        return float(csum)

    def step(self, x: HingeState, dt: float) -> Tuple[float, float, float]:
        # setpoint ramp
        sp, _ = Setpoint(self._sp, self._goal, self._sp_speed).step(dt)
        self._sp = sp
        e_now = sp - x.theta

        # integral bias (clamped)
        self.bias_u += self.Ki_bias * e_now * dt
        self.bias_u = max(-self.bias_limit, min(self.bias_u, self.bias_limit))

        # update acceleration and τ̂_f from torque residual (last_u - τ_dyn)
        a_raw = (x.omega - self._prev_v) / dt
        self._prev_v = x.omega
        self._a_hat = 0.2 * a_raw + 0.8 * self._a_hat
        tau_dyn = self.J * self._a_hat + self.D * x.omega
        tau_f_inst = self.last_u - tau_dyn          # includes friction + well (acts like a constant local offset)
        self.tau_f_est = (1.0 - self.ff_alpha) * self.tau_f_est + self.ff_alpha * tau_f_inst
        self.tau_f_est = max(-self.ff_clip, min(self.tau_f_est, self.ff_clip))

        # stuck detection & brief kick
        stuck_now = (abs(x.omega) < self.v_small) and (abs(e_now) > self.e_break)
        if self._kick_left > 0.0 or (stuck_now and (self.stuck_t + dt) >= self.stuck_thresh_s):
            # start/continue a short saturated pulse toward the goal
            self._kick_left = max(self._kick_left, self.kick_dur_s)
            u_cmd = math.copysign(self.tau_sat, e_now)
            self._kick_left = max(0.0, self._kick_left - dt)
            self.last_u = u_cmd
            # decay the stuck timer while kicking so we don't chain kicks
            self.stuck_t = max(0.0, self.stuck_t - 2*dt)
            return u_cmd, sp, self._goal

        # maintain a simple stuck timer when not kicking
        if stuck_now:
            self.stuck_t += dt
        else:
            self.stuck_t = max(0.0, self.stuck_t - 2*dt)

        # constant-input MPC with friction/disturbance offset + breakaway encouragement
        best_u, best_c = 0.0, float("inf")
        tau_break = max(0.5 * abs(self.tau_f_est), 0.25 * self.tau_sat)
        for u in ACTIONS:
            u = float(u)
            c = self._rollout_cost_const_u(x.theta, x.omega, u, sp, self.last_u, self.tau_f_est, dt)
            if stuck_now and (abs(u) < tau_break):
                c += self.r_break * (tau_break - abs(u))
            if c < best_c:
                best_c, best_u = c, u

        # add bias, saturate, return
        u_raw = best_u + self.bias_u
        u_cmd = max(-self.tau_sat, min(self.tau_sat, u_raw))
        self.last_u = u_cmd
        return u_cmd, sp, self._goal


class POMDPCtrl:
    """
    Very small POMDP: belief over friction scale α∈{0.85,1.0,1.15} that multiplies (Fs,Fc).
    Action = argmin over discrete grid of expected discounted cost under belief,
    with risk on saturation and large error. Belief updated using torque-residual
    likelihood in torque units (uses filtered acceleration).
    """
    def __init__(self, J=J_EQ, D=D_EQ, tau_sat=TAU_SAT, sp_speed=SP_SPEED):
        self.J, self.D = float(J), float(D)
        self.tau_sat = float(tau_sat)
        self._sp = 0.0; self._goal = 0.0; self._sp_speed = float(sp_speed)
        self.last_u = 0.0
        self.b = np.ones(len(FRIC_SCALES)) / len(FRIC_SCALES)
        self._a_hat = 0.0
        self._v_prev = 0.0

    def reset(self, q0: float, v0: float = 0.0):
        self._sp = self._goal = float(q0)
        self.last_u = 0.0
        self.b[:] = 1.0/len(FRIC_SCALES)
        self._v_prev = float(v0); self._a_hat = 0.0

    def set_goal(self, q_goal: float):
        self._goal = float(q_goal)

    def _expected_cost(self, x: HingeState, sp: float, dt: float, u: float) -> float:
        # one-step lookahead with short rollout using expected friction torque
        th, w = x.theta, x.omega
        cost = W_DU * (u - self.last_u)**2
        for h in range(MPC_H):  # reuse horizon/weights
            # expected friction torque at current w under belief
            Fs_b = FS * np.dot(self.b, FRIC_SCALES)
            Fc_b = FC * np.dot(self.b, FRIC_SCALES)
            tf = tau_fric_smooth(w, Fs_b, Fc_b)
            tw = tau_well(th, K=self._K_for_cost)  # set externally
            a = (u - self.D*w - tf - tw) / self.J
            w = w + dt*a
            th = th + dt*w
            e = th - sp
            w_disc = (MPC_GAMMA**h)
            # tracking + energy + soft L1 + gentle risk on action & error
            r_act = 1.0 if abs(u) > TAU_MAX_FOR_RISK else 0.0
            r_err = 1.0 if abs(e) > ERR_RISK_THRESH else 0.0
            cost += w_disc*(W_TH*(e*e) + W_W*(w*w) + W_U*(u*u) + W_ERR_SOFT*abs(e) +
                            0.2*W_RISK_ACT*r_act + 0.2*W_RISK_ERR*r_err)
        return float(cost)

    def step(self, x: HingeState, dt: float, K_for_cost: float) -> Tuple[float, float, float]:
        # setpoint ramp
        sp, _ = Setpoint(self._sp, self._goal, self._sp_speed).step(dt)
        self._sp = sp
        self._K_for_cost = float(K_for_cost)

        # choose action
        best_u, best_c = 0.0, float('inf')
        for u in ACTIONS:
            c = self._expected_cost(x, sp, dt, float(u))
            if c < best_c:
                best_c, best_u = c, float(u)
        u_cmd = max(-self.tau_sat, min(self.tau_sat, best_u))
        self.last_u = u_cmd
        return u_cmd, sp, self._goal

    def observe_and_update(self, x: HingeState, dt: float, u_applied: float, K_true: float):
        # torque residual likelihood in torque units
        a_raw = (x.omega - self._v_prev) / dt
        self._v_prev = x.omega
        self._a_hat = 0.2*a_raw + 0.8*self._a_hat
        tau_dyn = self.J*self._a_hat + self.D*x.omega
        # residual vs. friction+well predictions per friction scale
        e = []
        for alpha in FRIC_SCALES:
            tf = tau_fric_smooth(x.omega, Fs=alpha*FS, Fc=alpha*FC)
            tw = tau_well(x.theta, K_true)
            e.append( (u_applied - tau_dyn) - (tf + tw) )
        e = np.array(e)
        L = np.exp(-0.5*(e / max(SIGMA_RESIDUAL,1e-6))**2)
        L = np.clip(L, 1e-12, None)
        b_post = self.b * L
        b_post = np.clip(b_post, 1e-6, None)
        b_post /= b_post.sum()
        # small stickiness
        self.b = 0.95*b_post + 0.05*(np.ones_like(b_post)/len(b_post))

# ------------------------------- Simulation ----------------------------------

def deg(x): return x*180.0/math.pi

def escape_time(t, angle, setpoint, goal):
    """Primary: within ESC_TOL_DEG for ESC_DWELL_S; Fallback: 4%% pairwise closeness & non-zero."""
    tol = math.radians(ESC_TOL_DEG)
    inside = np.abs(angle - goal) <= tol
    if inside.any():
        dt = np.diff(t, prepend=t[0])
        dwell = 0.0
        for i in range(len(t)):
            dwell = dwell + dt[i] if inside[i] else 0.0
            if dwell >= ESC_DWELL_S:
                # backtrack to segment entry
                j = i
                while j > 0 and inside[j-1]:
                    j -= 1
                return float(t[j])
    # fallback: all non-zero and within 4% pairwise
    pct = 0.04; eps = 1e-9
    triple = np.vstack([angle, setpoint, goal*np.ones_like(angle)]).T
    nonzero = (np.abs(triple) > eps).all(axis=1)
    maxi = np.max(np.abs(triple), axis=1)
    spread = np.max(np.abs(triple[:,[0,0,1]] - triple[:,[1,2,2]]), axis=1)  # (|a-sp|,|a-g|,|sp-g|)
    close = spread <= pct * np.maximum(maxi, eps)
    idx = np.argmax(nonzero & close)
    if (nonzero & close).any():
        return float(t[idx])
    return float('nan')

def simulate_once(ctrl_name: str, profile: str, seed: int = 0, noise_std: float = NOISE_STD) -> Tuple[pd.DataFrame, Dict[str,float]]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed); random.seed(seed)

    K = PROFILES[profile]["K"]
    params = PlantParams(J=J_EQ, D=D_EQ, Fs=FS, Fc=FC, K=K, n=WELL_N)
    plant = HingePlant(params, noise_std=noise_std)

    # controllers
    if ctrl_name == "PID":
        ctrl = PIDCtrl(PID_KP, PID_KI, PID_KD, tau_sat=TAU_SAT, i_frac=PID_I_FRAC, sp_speed=SP_SPEED)
        ctrl.reset(0.0)
    elif ctrl_name == "MPC":
        ctrl = MPCCtrl(J_EQ, D_EQ, tau_sat=TAU_SAT, sp_speed=SP_SPEED)
        ctrl.reset(0.0)
    elif ctrl_name == "POMDP":
        ctrl = POMDPCtrl(J_EQ, D_EQ, tau_sat=TAU_SAT, sp_speed=SP_SPEED)
        ctrl.reset(0.0, 0.0)
    else:
        raise ValueError("Unknown controller")

    # set goal
    ctrl.set_goal(math.radians(SP_STEP_DEG))

    # state
    x = HingeState(theta=0.0, omega=0.0)

    # logging buffers
    rows = []
    t = 0.0; N = int(T_TOTAL/DT)
    tau_load = 0.0  # Stage A has no external load; keep column for compatibility

    # simple low-pass of acceleration for PID/MPC telemetry
    a_hat = 0.0; v_prev = 0.0

    for k in range(N):
        # controller step
        if ctrl_name == "POMDP":
            u, sp, sp_goal = ctrl.step(x, DT, K_for_cost=K)
        else:
            u, sp, sp_goal = ctrl.step(x, DT)

        # apply to plant
        x_next, diag = plant.step(x, u, DT)

        # risk surrogates (for compatibility)
        risk_act = 1.0 if abs(u) > TAU_MAX_FOR_RISK else 0.0
        risk_err = 1.0 if abs(sp - x.theta) > ERR_RISK_THRESH else 0.0
        risk = max(risk_act, risk_err)

        # accel estimate
        a_raw = (x.omega - v_prev)/DT
        v_prev = x.omega
        a_hat = 0.2*a_raw + 0.8*a_hat

        rows.append(dict(
            t=t,
            angle=x.theta, setpoint=sp, sp_goal=sp_goal,
            dq=x.omega, a_hat=a_hat,
            tau_dyn=J_EQ*a_hat + D_EQ*x.omega,
            tau_cmd=u, tau_load=tau_load,
            tau_f_exp=0.0,   # not used here
            bias_u=(ctrl.I*PID_KI if ctrl_name=="PID" else 0.0),
            Fx=0.0, Fy=0.0, Fz=0.0, Mx=0.0, My=0.0, Mz=0.0,
            pos_err=0.0, rot_err=0.0,
            risk_act=risk_act, risk_err=risk_err, risk=risk,
            J_est=J_EQ, D_est=D_EQ,
            # POMDP belief (pad to 5 for compatibility with prior)
            b0=(ctrl.b[0] if ctrl_name=="POMDP" else 0.0),
            b1=(ctrl.b[1] if ctrl_name=="POMDP" else 0.0),
            b2=(ctrl.b[2] if ctrl_name=="POMDP" else 0.0),
            b3=0.0, b4=0.0
        ))

        # update POMDP belief using actual torque residual
        if ctrl_name == "POMDP":
            ctrl.observe_and_update(x, DT, u_applied=u, K_true=K)

        # advance
        x = x_next
        t += DT

    df = pd.DataFrame(rows)

    # metrics
    tesc = escape_time(df["t"].values, df["angle"].values, df["setpoint"].values, df["sp_goal"].iloc[-1])
    w_mech = float(np.trapz(np.abs(df["tau_cmd"].values * df["dq"].values), df["t"].values))
    # overshoot (in direction of motion)
    dir_sign = np.sign(df["sp_goal"].iloc[-1] - df["angle"].iloc[0] + 1e-12)
    overs = dir_sign * (df["angle"].values - df["sp_goal"].iloc[-1])
    theta_os = float(np.maximum(0.0, overs).max())
    # saturation time (ms)
    sat_mask = np.abs(df["tau_cmd"].values) >= SAT_FRAC*TAU_SAT
    dt = np.diff(df["t"].values, prepend=df["t"].values[0])
    sat_ms = float(1000.0 * np.sum(dt[sat_mask]))

    stats = dict(t_esc_s=tesc, W_mech_J=w_mech, theta_os_rad=theta_os, Sat_time_ms=sat_ms)
    return df, stats

# ---------------------------- Experiment driver ------------------------------

def chi2_95():
    # 95% quantile for chi^2 with 2 DOF (ellipse scale)
    return 5.991

def ellipse_points(mu: np.ndarray, cov: np.ndarray, n=200) -> np.ndarray:
    # returns (n,2) points forming a 95% ellipse around mu
    if not np.all(np.isfinite(cov)) or cov.shape != (2,2):
        cov = np.eye(2)*1e-6
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    L = np.diag(np.sqrt(vals * chi2_95()))
    A = vecs @ L
    angles = np.linspace(0, 2*np.pi, n)
    circle = np.vstack([np.cos(angles), np.sin(angles)])
    pts = (A @ circle).T + mu[None,:]
    return pts

def run_all(trials=10, seed=0, noise=NOISE_STD):
    rng = np.random.default_rng(seed)
    combos = [(c,p) for p in PROFILES.keys() for c in ("PID","MPC","POMDP")]
    summary_rows = []
    medians = {}      # (c,p) -> (t_med, w_med)
    boots = {}        # (c,p) -> boot samples (B,2)
    # per-controller representative trace (Nominal)
    reps = {}

    for p in PROFILES.keys():
        for c in ("PID","MPC","POMDP"):
            t_esc_list, w_mech_list = [], []
            # representative trace: first trial for Nominal
            rep_df = None

            for k in range(trials):
                df, stats = simulate_once(c, p, seed=rng.integers(0, 10**9).item(), noise_std=noise)
                if (p == "Nominal") and (rep_df is None):
                    reps[c] = df.copy()
                # write CSV for last trial only?  → Write the first (deterministic seed) AND overwrite after loop with the median trial.
                # To keep things simple and consistent with downstream scripts, write the LAST run:
                fname = f"{c}-{p}.csv"
                df.to_csv(os.path.join(".", fname), index=False)

                t_esc_list.append(stats["t_esc_s"])
                w_mech_list.append(stats["W_mech_J"])

            t_med = float(np.nanmedian(t_esc_list))
            w_med = float(np.nanmedian(w_mech_list))
            medians[(c,p)] = (t_med, w_med)

            # bootstrap medians
            B = 600
            boot = []
            idxs = np.arange(len(t_esc_list))
            for _ in range(B):
                ii = np.random.choice(idxs, size=len(idxs), replace=True)
                boot.append([np.nanmedian(np.array(t_esc_list)[ii]),
                             np.nanmedian(np.array(w_mech_list)[ii])])
            boots[(c,p)] = np.array(boot)

            # Save summary row
            summary_rows.append(dict(
                Profile=p, Controller=("MPC/MPPI" if c=="MPC" else c),
                t_esc_s=t_med, W_mech_J=w_med,
                theta_os_rad=np.nan, Sat_time_ms=np.nan
            ))

    # Make figures
    fig_trade = plt.figure(figsize=(8.2, 6.2))
    ax = fig_trade.add_subplot(111)
    colors = dict(PID="#1f77b4", MPC="#ff7f0e", POMDP="#2ca02c")
    markers = dict(Light="o", Nominal="s", Severe="^")
    for (c,p), (t_med, w_med) in medians.items():
        ax.scatter(t_med, w_med, c=colors[c], marker=markers[p], s=70, label=f"{c}-{p}")
        # 95% boot ellipse
        pts = ellipse_points(np.array([t_med, w_med]), np.cov(boots[(c,p)].T))
        ax.plot(pts[:,0], pts[:,1], c=colors[c], alpha=0.55)
    # legend: group by controller (color) & profile (marker)
    from matplotlib.lines import Line2D
    lc = [Line2D([0],[0], color=colors[c], lw=2) for c in ("PID","MPC","POMDP")]
    lm = [Line2D([0],[0], marker=markers[p], color='k', lw=0, markerfacecolor='none', markersize=8) for p in PROFILES.keys()]
    leg1 = ax.legend(lc, ["PID","MPC/MPPI","POMDP"], loc="upper right", title="Controller")
    ax.add_artist(leg1)
    ax.legend(lm, list(PROFILES.keys()), loc="lower right", title="Profile")
    ax.set_xlabel(r"$t_{\mathrm{esc}}$ (s)")
    ax.set_ylabel(r"$W_{\mathrm{mech}}$ (J)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Stage A (Python): time–work trade‑off")
    trade_path = os.path.join(OUT_DIR, "time_work_tradeoff_python.pdf")
    fig_trade.tight_layout()
    fig_trade.savefig(trade_path, bbox_inches="tight")
    plt.close(fig_trade)

    # State traces under Nominal
    fig_tr = plt.figure(figsize=(9.5, 5.8))
    axs = [fig_tr.add_subplot(3,1,i+1) for i in range(3)]
    for c in ("PID","MPC","POMDP"):
        df = reps[c]
        axs[0].plot(df["t"], df["angle"], label=f"{c}")
        axs[0].plot(df["t"], df["setpoint"], lw=1, alpha=0.7, ls="--", c="k")
        axs[1].plot(df["t"], df["dq"], label=f"{c}")
        axs[2].plot(df["t"], df["tau_cmd"], label=f"{c}")
    axs[0].set_ylabel(r"$\theta$ (rad)"); axs[1].set_ylabel(r"$\dot\theta$ (rad/s)"); axs[2].set_ylabel(r"$\tau$ (N·m)")
    axs[2].set_xlabel("time (s)")
    for ax in axs:
        ax.grid(True, alpha=0.3); ax.set_xlim(0, T_TOTAL)
    axs[0].set_title("Stage A (Python): representative traces (Nominal)")
    axs[0].legend(loc="best")
    trace_path = os.path.join(OUT_DIR, "state_traces_python.pdf")
    fig_tr.tight_layout()
    fig_tr.savefig(trace_path, bbox_inches="tight")
    plt.close(fig_tr)

    # Write Stage A summary table (CSV + LaTeX)
    summary = pd.DataFrame(summary_rows)
    # Fill overshoot/sat if desired (kept NaN in this simple run)
    summary.to_csv("stageA_summary.csv", index=False)
    write_latex_table(summary, "stageA_summary.tex")
    print(f"[OK] Wrote:\n  {trade_path}\n  {trace_path}\n  ./stageA_summary.csv\n  ./stageA_summary.tex\n  + 9 CSV logs in current dir")

def write_latex_table(df: pd.DataFrame, path: str):
    # Arrange rows by profile with blanked repeats of profile label like your excerpt
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Stage A summary}")
    lines.append(r"  \label{tab:stageA_summary}")
    lines.append(r"  \vspace{4pt}")
    lines.append(r"  \begin{tabular}{l l c c c c}")
    lines.append(r"    \toprule")
    lines.append(r"    Profile & Controller & $t_{\mathrm{esc}}$ (s) & $W_{\mathrm{mech}}$ (J) & $\theta_{\mathrm{os}}$ (rad) & Sat.\ time (ms)\\")
    lines.append(r"    \midrule")
    for prof in ["Light","Nominal","Severe"]:
        rows = df[df["Profile"]==prof]
        for j, (_, r) in enumerate(rows.iterrows()):
            ptxt = prof if j==0 else " "
            ctrl = r["Controller"]
            te = f"{r['t_esc_s']:.3f}" if np.isfinite(r['t_esc_s']) else ""
            wm = f"{r['W_mech_J']:.3f}" if np.isfinite(r['W_mech_J']) else ""
            os_ = f"{r['theta_os_rad']:.3f}" if np.isfinite(r['theta_os_rad']) else ""
            st_ = f"{r['Sat_time_ms']:.0f}" if np.isfinite(r['Sat_time_ms']) else ""
            lines.append(f"    {ptxt} & {ctrl} & {te} & {wm} & {os_} & {st_} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ---------------------------------- CLI --------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10, help="trials per (controller,profile)")
    ap.add_argument("--seed",   type=int, default=0,  help="random seed")
    ap.add_argument("--noise",  type=float, default=NOISE_STD, help="process noise std on θ̈ (rad/s^2)")
    args = ap.parse_args()

    np.random.seed(args.seed); random.seed(args.seed)
    run_all(trials=args.trials, seed=args.seed, noise=args.noise)

if __name__ == "__main__":
    main()
