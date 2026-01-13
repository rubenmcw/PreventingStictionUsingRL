#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR10e + Barrett hand + panel hinge (IMPROVED):
----------------------------------------------
This version folds in the following improvements:

(1) Correct Bayesian observation model (residuals in **torque** units) with a
    light low‑pass on the finite‑difference acceleration estimate.
(2) Correct hinge inertia: compute the equivalent generalized inertia J_eq from
    MuJoCo’s mass matrix at runtime and use it for the horizon model and the
    likelihood (fallback default 0.08 kg·m² if computation isn’t available).
(3) Risk term penalizes **actuator saturation** (|u| > tau_max) rather than
    “net hinge torque,” removing bias against applying needed torque.
(5) Replace output low‑pass with a gentle **Δu penalty** in the short‑horizon
    cost (no added lag at steady‑state).
(6) Densify discrete action set near zero torque to reduce dithering.
(7) Add a small, bounded **integral bias** term to remove steady‑state error.

Notes:
• The feed‑forward of the synthetic sinusoidal load into the likelihood (4),
  ramp‑consistent horizon reference (8), qfrc_applied clearing (9), contact/base
  gain retuning (10), UR passive damping scaling (11), and QoL items (12) are
  intentionally **not** included per request.
"""

from __future__ import annotations

import sys
import os
import time
import math
import threading
import queue
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

import mujoco as mj
from mujoco import mjtObj

from pomdp_optimizer import DeltaUOptimizerConfig, DeltaUSequenceOptimizer

# =============================================================================
# ------------------------------- CONFIG --------------------------------------
# =============================================================================

# --- time base
TOTAL_TIME_S: float = 25.0
CTRL_DT: float      = 1.0 / 200.0

# --- approach / alignment
EDGE_BODY_PREFERENCE = ["flagp_wam", "flagp", "flagp_ur5e", "hinge_link"]
PREGRASP_DIST        = 0.14
APPROACH_DIST        = 0.02
VERT_OFFSET_BELOW_M  = 0.03

EDGE_AXIS            = "x"     # 'x' or 'y' edge of panel to align to
PAIR_AXIS_LOCAL      = "x"     # Barrett pair axis in palm frame

YAW_KP               = 1.4
YAW_KD               = 0.40
YAW_MAX_STEP         = math.radians(6.0)
YAW_TOL              = math.radians(1.0)
YAW_TOTAL_CAP        = math.radians(66.0)
YAW_ALIGN_TIMEOUT_S  = 10.0
YAW_DIR_SIGN         = -1

# --- joint freeze & settling
FREEZE_PANEL_JOINT   = "hinge_joint"
FREEZE_WAM_JOINTS    = [
    "wam/joints/base_yaw",
    "wam/joints/shoulder_pitch_joint",
    "wam/joints/shoulder_yaw",
    "wam/joints/elbow_pitch",
    "wam/joints/wrist_yaw",
    "wam/joints/wrist_pitch",
    "wam/joints/palm_yaw",
]
WAM_FREEZE_DELAY_S   = 3.0
POST_LATCH_SETTLE_S  = 0.6

# --- base rigid hold
BASE_JOINTS_TO_FREEZE = ["fix_roll","fix_pitch","fix_yaw","fix_z","fix_y","fix_x"]
BASE_BODY_NAME        = "EE_sate"
BASE_VH_POS_KP        = 8000.0
BASE_VH_POS_KD        = 220.0
BASE_VH_ROT_KP        = 450.0
BASE_VH_ROT_KD        = 22.0

# --- UR passive after latch
UR_PASSIVE_AFTER_LATCH   = True
UR_DAMP_SCALE_ON_PASSIVE = 1.0  # unchanged (see note at top)

# --- hand close after latch
PALM_BODY_NAME       = "wam/bhand/bhand_palm_link"
TIP_CLOSE_FRAC       = 0.77

# --- equality toggles
DISABLE_EQ_FIXER_AT_LOAD = True
FIXER_EQ_NAME            = "fixer"   # if present in XML (may be absent)

# --- live plotting
PLOT_RATE_HZ        = 10.0   # UI timer rate
PLOT_QUEUE_SIZE     = 2000
PLOT_WINDOW_SEC     = 30.0   # dynamic autoscale over whole trajectory anyway

# --- optional telemetry persistence
SAVE_CSV            = False
SAVE_NPZ            = False
SAVE_DIR            = "telemetry_out"

# --- POMDP hinge config -------------------------------------------------------
POMDP_TAU_SAT            = 5.0                 # torque cap (N·m)
POMDP_SP_SPEED_RAD_S     = math.radians(12.0)  # setpoint slew

@dataclass
class StribeckParameters:
    Fc: float
    Fs: float

# friction magnitudes scaled down vs. torque cap
POMDP_FRIC = StribeckParameters(Fc=0.35 * POMDP_TAU_SAT,
                                Fs=0.60 * POMDP_TAU_SAT)

# (1) Correct likelihood units → residual noise now in **torque** units
SIGMA_RESIDUAL      = 0.30         # N·m residual noise scale
BELIEF_FLOOR        = 1e-6
P_MAX_RISK          = 0.4
TAU_MAX_FOR_RISK    = 1.3 * POMDP_TAU_SAT
N_ACTIONS           = 257
# (6) Densify action grid near zero by warping a uniform grid
_z = np.linspace(-1.0, 1.0, N_ACTIONS)
ACTIONS             = POMDP_TAU_SAT * np.sign(_z) * (np.abs(_z)**1.7)
POMDP_CTRL_MODE     = "cem"  # "discrete", "cem", or "mppi"

# Continuous action optimization (CEM / MPPI)
POMDP_CEM_SAMPLES   = 192
POMDP_CEM_ELITES    = 24
POMDP_CEM_ITERS     = 4
POMDP_CEM_INIT_STD  = 0.6 * POMDP_TAU_SAT
POMDP_MPPI_SAMPLES  = 256
POMDP_MPPI_LAMBDA   = 6.0
POMDP_MPPI_SIGMA    = 0.5 * POMDP_TAU_SAT

# POMDP “model” / cost weights
# (2) J will be computed from the model at runtime; this is a safe default
POMDP_J_EST        = 0.08
POMDP_D_EST        = 0.8
POMDP_W_TRACK      = 50.0
POMDP_W_VEL        = 0.5
POMDP_W_U          = 8e-5
POMDP_W_RISK       = 1.5
# (5) Δu penalty replaces output low‑pass smoothing
POMDP_W_DU         = 2e-4
POMDP_U_SMOOTH_ALPHA = 0.0  # keep path but disabled
# Explicit saturation penalties
POMDP_W_SAT        = 20.0
POMDP_W_RAIL       = 5.0
POMDP_SAT_ALPHA    = 0.8
POMDP_RAIL_BETA    = 0.95
POMDP_DU_MAX       = 0.10 * POMDP_TAU_SAT

# Short horizon look‑ahead for POMDP action evaluation (keeps runtime light)
POMDP_H_CTRL       = 12
POMDP_GAMMA        = 0.98

# Sinusoidal resistance injected at the hinge (scaled)
LOAD_SINE_FREQ_HZ   = 0.5
LOAD_SINE_AMP_FRAC  = 0.35  # of POMDP_TAU_SAT
LOAD_PHASE          = 0.0


# =============================================================================
# ------------------------------- UTILITIES -----------------------------------
# =============================================================================

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def id_or_fail(model, objtype, name):
    i = mj.mj_name2id(model, objtype, name)
    if i < 0:
        raise RuntimeError(f"{name!r} not found.")
    return i

# --- quaternion helpers
def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_conj(q):
    w,x,y,z = q
    return np.array([w, -x, -y, -z], dtype=float)

def normalize_quat(q):
    n = np.linalg.norm(q)
    return q if n == 0 else q / n

def quat_to_mat(q):
    w,x,y,z = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=float)

def quat_error_small(q_des, q_cur):
    """Small-angle rotation vector from q_des → q_cur (world frame)."""
    q_err = quat_mul(quat_conj(q_des), q_cur)  # des^-1 * cur
    q_err = normalize_quat(q_err)
    w, x, y, z = q_err
    signw = 1.0 if w >= 0.0 else -1.0
    return 2.0 * signw * np.array([x, y, z], dtype=float)

def world_body_vel(model, data, body_id: int):
    Jp = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
    mj.mj_jacBody(model, data, Jp, Jr, body_id)
    qvel = np.asarray(data.qvel, dtype=float)
    vlin = Jp @ qvel
    vang = Jr @ qvel
    return vlin, vang

def hinge_equivalent_inertia(model, data, joint_name: str) -> float:
    """
    (2) Compute the equivalent generalized inertia at a 1‑DoF hinge by taking
    the diagonal element of M(q) corresponding to that DOF.
    """
    jid = mj.mj_name2id(model, mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return POMDP_J_EST
    did = model.jnt_dofadr[jid]
    v = np.zeros(model.nv); v[did] = 1.0
    Mv = np.zeros(model.nv)
    mj.mj_mulM(model, data, Mv, v)
    return float(Mv[did])


# =============================================================================
# ---------------------------- LOW-LEVEL PD -----------------------------------
# =============================================================================

class JointPD:
    """Simple joint PD mapped to MuJoCo actuators by name."""
    def __init__(self, model, data, joint_names: List[str], act_names: List[str],
                 kp: List[float], kd: List[float], tau_limit: List[float]):
        self.model = model; self.data = data
        self.jids = [id_or_fail(model, mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        self.qadr = [model.jnt_qposadr[j] for j in self.jids]
        self.vadr = [model.jnt_dofadr[j]  for j in self.jids]
        self.aids = [id_or_fail(model, mjtObj.mjOBJ_ACTUATOR, n) for n in act_names]
        self.kp = np.asarray(kp, float); self.kd = np.asarray(kd, float)
        self.tau_lim = np.asarray(tau_limit, float)
        self.q_des = np.array([data.qpos[a] for a in self.qadr], float)
        self.enabled = True

    def set_q_des(self, qd: np.ndarray):
        self.q_des = np.asarray(qd, float)

    def hold_here(self):
        self.q_des = np.array([self.data.qpos[a] for a in self.qadr], float)

    def set_enabled(self, on: bool):
        self.enabled = bool(on)

    def step(self):
        if not self.enabled:
            for aid in self.aids: self.data.ctrl[aid] = 0.0
            return
        q  = np.array([self.data.qpos[a] for a in self.qadr], float)
        dq = np.array([self.data.qvel[a] for a in self.vadr], float)
        tau = self.kp*(self.q_des - q) - self.kd*dq
        tau = np.clip(tau, -self.tau_lim, self.tau_lim)
        for i, aid in enumerate(self.aids):
            self.data.ctrl[aid] = float(tau[i])


class UR10:
    """UR10 PD with basic IK to a point and joint‑6 yaw alignment utility."""
    def __init__(self, model, data):
        self.model = model; self.data = data
        self.jnames = ["ur10_joint_1","ur10_joint_2","ur10_joint_3",
                       "ur10_joint_4","ur10_joint_5","ur10_joint_6"]
        self.servo = JointPD(model, data, self.jnames,
                             ["ur10_shoulder_pan","ur10_shoulder_lift","ur10_elbow",
                              "ur10_wrist_1","ur10_wrist_2","ur10_wrist_3"],
                             kp=[300,300,250,120,80,60],
                             kd=[ 25, 25, 20,  6, 5, 3],
                             tau_limit=[120,120,100,50,35,25])
        self.jids = self.servo.jids
        self.qadr = self.servo.qadr
        self.site_id = id_or_fail(model, mjtObj.mjOBJ_SITE, "ur10_attachment_site")
        self.palm_bid = id_or_fail(model, mjtObj.mjOBJ_BODY, PALM_BODY_NAME)
        self.Jp = np.zeros((3, model.nv)); self.Jr = np.zeros((3, model.nv))
        self.cols = []
        for j in self.jids:
            adr = model.jnt_dofadr[j]
            jtype = self.model.jnt_type[j]
            nv = 6 if jtype == mj.mjtJoint.mjJNT_FREE else 3 if jtype == mj.mjtJoint.mjJNT_BALL else 1
            self.cols.extend(range(adr, adr+nv))
        self.freeze_all = False
        self.j6_only    = False
        self.q6_start   = None

    def _q(self):  return np.array([self.data.qpos[a] for a in self.qadr], float)
    def _set(self, q): self.servo.set_q_des(q)
    def freeze_pose(self): self.freeze_all = True; self.j6_only = False; self.servo.hold_here()
    def set_passive(self, passive: bool): self.servo.set_enabled(not passive)

    def scale_damping(self, scale: float):
        scale = float(scale)
        for dof in self.cols:
            self.model.dof_damping[dof] *= scale

    def enable_joint6_only(self):
        self.freeze_all = False; self.j6_only = True; self.servo.hold_here()
        self.q6_start = float(self.data.qpos[self.qadr[-1]])

    def ik_to_point(self, p_des: np.ndarray, step_scale: float = 0.65, damping: float = 1e-3):
        if self.freeze_all or self.j6_only: return
        p_cur = np.array(self.data.site_xpos[self.site_id], float)
        e = p_des - p_cur
        mj.mj_jacSite(self.model, self.data, self.Jp, self.Jr, self.site_id)
        J = self.Jp[:, self.cols]
        A = J @ J.T + damping*np.eye(3)
        dq = J.T @ np.linalg.solve(A, e)
        q = self._q(); q_des = q + step_scale * dq
        for i, jid in enumerate(self.jids):
            if self.model.jnt_limited[jid] == 1:
                lo, hi = self.model.jnt_range[jid]; q_des[i] = clamp(q_des[i], lo, hi)
        self._set(q_des)

    def yaw_align_step_joint6(self, desired_world_dir: np.ndarray):
        if not self.j6_only: return None
        R = np.array(self.data.xmat[self.palm_bid], float).reshape(3,3)
        pair_axis = R[:,0] if PAIR_AXIS_LOCAL.lower().startswith("x") else R[:,1]
        def unit2(v):
            u = np.array([v[0], v[1]], float); n = np.linalg.norm(u)
            return u/(n + 1e-12)
        a = unit2(pair_axis); b = unit2(desired_world_dir)
        crossz = a[0]*b[1] - a[1]*b[0]
        dotab  = a[0]*b[0] + a[1]*b[1]
        err = math.atan2(crossz, dotab)
        if abs(err) > math.pi/2:
            err -= math.copysign(math.pi, err)
        jid6 = self.jids[-1]; vadr6 = self.model.jnt_dofadr[jid6]
        q6dot = float(self.data.qvel[vadr6])
        step = (YAW_KP*err - YAW_KD*q6dot) * YAW_DIR_SIGN
        step = clamp(step, -YAW_MAX_STEP, YAW_MAX_STEP)
        step = math.copysign(min(abs(step), 0.5*abs(err)), step)
        q = self._q(); q6_new = q[-1] + step
        if self.q6_start is not None and abs(q6_new - self.q6_start) > YAW_TOTAL_CAP:
            q6_new = self.q6_start + math.copysign(YAW_TOTAL_CAP, q6_new - self.q6_start)
        if self.model.jnt_limited[jid6] == 1:
            lo, hi = self.model.jnt_range[jid6]; q6_new = clamp(q6_new, lo, hi)
        q[-1] = q6_new
        self._set(q)
        return err

    def step_pd(self): self.servo.step()


class BarrettHand:
    """Barrett hand controller: optional tip‑only close (no proximal rotation)."""
    def __init__(self, model, data):
        self.model = model; self.data = data
        self.jnames = [
            "wam/bhand/finger_3/med_joint","wam/bhand/finger_3/dist_joint",
            "wam/bhand/finger_1/prox_joint","wam/bhand/finger_1/med_joint","wam/bhand/finger_1/dist_joint",
            "wam/bhand/finger_2/prox_joint","wam/bhand/finger_2/med_joint","wam/bhand/finger_2/dist_joint",
        ]
        self.anames = ["finger_3_1","finger_3_2","finger_1_1","finger_1_2","finger_1_3",
                       "finger_2_1","finger_2_2","finger_2_3"]
        self.servo = JointPD(model, data, self.jnames, self.anames,
                             kp=[15,12, 18,12,10, 18,12,10],
                             kd=[1.5,1.0, 2.0,1.2,1.0, 2.0,1.2,1.0],
                             tau_limit=[2.0]*8)
        self.jids = self.servo.jids
        self.idx_f1_prox = 2
        self.idx_f2_prox = 5

    def open(self):
        q = []
        for jid in self.jids:
            lo, hi = (0.0, 0.0)
            if self.model.jnt_limited[jid] == 1: lo, hi = self.model.jnt_range[jid]
            q.append(max(lo, 0.0))
        self.servo.set_q_des(np.array(q, float))

    def close_partial_no_rotation(self, frac: float = TIP_CLOSE_FRAC):
        frac = float(np.clip(frac, 0.0, 1.0))
        q_des = []
        for i, jid in enumerate(self.jids):
            if i in (self.idx_f1_prox, self.idx_f2_prox):
                q_des.append(float(self.data.qpos[self.servo.qadr[i]]))  # lock rotation
            else:
                if self.model.jnt_limited[jid] == 1:
                    lo, hi = self.model.jnt_range[jid]; q_des.append(lo + frac*(hi - lo))
                else:
                    q_des.append(frac)
        self.servo.set_q_des(np.array(q_des, float))

    def step_pd(self): self.servo.step()


# =============================================================================
# ------------------------------- FREEZERS ------------------------------------
# =============================================================================

class JointFreezer:
    """Holds a single named joint (qpos/qvel)."""
    def __init__(self, model, data, joint_name: str):
        self.model = model; self.data = data
        jid = mj.mj_name2id(model, mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            print(f"[freezer] WARN: joint '{joint_name}' not found; disabled.")
            self.enabled = False; self.qadr = 0; self.vadr = 0; self.nv = 0; self.qref = np.zeros(0)
            return
        self.enabled = True
        self.qadr = model.jnt_qposadr[jid]; self.vadr = model.jnt_dofadr[jid]
        jtype = model.jnt_type[jid]
        self.nv = 6 if jtype == mj.mjtJoint.mjJNT_FREE else 3 if jtype == mj.mjtJoint.mjJNT_BALL else 1
        self.qref = np.array(data.qpos[self.qadr:self.qadr+self.nv], float)
        print(f"[freezer] Holding '{joint_name}' at qpos={self.qref.tolist()} (nv={self.nv})")

    def release(self): self.enabled = False

    def apply(self, data):
        if not self.enabled: return
        data.qpos[self.qadr:self.qadr+self.nv] = self.qref
        data.qvel[self.vadr:self.vadr+self.nv] = 0.0


class JointGroupFreezer:
    """Captures and holds a set of named joints."""
    def __init__(self, model, data, joint_names: List[str]):
        self.model = model; self.data = data
        self.items = []
        for name in joint_names:
            jid = mj.mj_name2id(model, mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                print(f"[group-freeze] WARN: joint '{name}' not found; skipping.")
                continue
            qadr = model.jnt_qposadr[jid]; vadr = model.jnt_dofadr[jid]
            jtype = model.jnt_type[jid]
            nv = 6 if jtype == mj.mjtJoint.mjJNT_FREE else 3 if jtype == mj.mjtJoint.mjJNT_BALL else 1
            self.items.append((qadr, vadr, nv, name))
        self.enabled = False; self.qrefs = None

    def enable(self, data):
        self.qrefs = [np.array(data.qpos[qadr:qadr+nv], float) for (qadr, _, nv, _) in self.items]
        self.enabled = True
        print(f"[group-freeze] ENABLED for {len(self.items)} joints.")

    def apply(self, data):
        if not self.enabled or not self.items: return
        for (qadr, vadr, nv, _), qref in zip(self.items, self.qrefs):
            data.qpos[qadr:qadr+nv] = qref
            data.qvel[vadr:vadr+nv] = 0.0


# =============================================================================
# -------------------------- LATCH & VIRTUAL CONTACT --------------------------
# =============================================================================

EQ_STRIDE = 7  # number of floats per equality row

def _find_latch_row(model, body1_name: str, body2_name: str) -> int:
    b1 = id_or_fail(model, mjtObj.mjOBJ_BODY, body1_name)
    b2 = id_or_fail(model, mjtObj.mjOBJ_BODY, body2_name)
    for i in range(model.neq):
        if model.eq_type[i] != mj.mjtEq.mjEQ_WELD:
            continue
        if model.eq_obj1id[i] == b1 and model.eq_obj2id[i] == b2:
            return i
    return -1

def _set_eq_data_row(model, eq_index: int, row7: np.ndarray):
    row7 = np.asarray(row7, dtype=float).reshape(7)
    try:
        model.eq_data[eq_index, :7] = row7
    except Exception:
        s = EQ_STRIDE * eq_index
        model.eq_data[s:s+EQ_STRIDE] = row7

def compute_relpose_in_body1_frame(data, body1_id: int, body2_id: int) -> np.ndarray:
    p1 = np.array(data.xpos[body1_id], float)
    p2 = np.array(data.xpos[body2_id], float)
    q1 = normalize_quat(np.array(data.xquat[body1_id], float))
    q2 = normalize_quat(np.array(data.xquat[body2_id], float))
    q12 = normalize_quat(quat_mul(quat_conj(q1), q2))
    R1 = quat_to_mat(q1)
    p12 = R1.T @ (p2 - p1)
    return np.array([p12[0], p12[1], p12[2], q12[0], q12[1], q12[2], q12[3]], dtype=float)

class LatchPair:
    """Keeps the equality row following until latch, then fixes it if available."""
    def __init__(self, model, data, body1="EE_ur10", body2="wam_grasp_point"):
        self.model = model; self.data = data
        self.body1_id = id_or_fail(model, mjtObj.mjOBJ_BODY, body1)
        self.body2_id = id_or_fail(model, mjtObj.mjOBJ_BODY, body2)
        self.eq_index = _find_latch_row(model, body1, body2)
        self.has_eq_active = hasattr(model, "eq_active")
        self.follow_mode = True
        if self.eq_index >= 0 and self.has_eq_active:
            try: model.eq_active[self.eq_index] = 0
            except Exception: pass
            mj.mj_forward(model, data)

    def follow_tick(self):
        if not self.follow_mode or self.eq_index < 0: return
        rel = compute_relpose_in_body1_frame(self.data, self.body1_id, self.body2_id)
        _set_eq_data_row(self.model, self.eq_index, rel)

    def latch_now(self):
        if self.eq_index >= 0:
            rel = compute_relpose_in_body1_frame(self.data, self.body1_id, self.body2_id)
            _set_eq_data_row(self.model, self.eq_index, rel)
        self.follow_mode = False
        if self.has_eq_active and self.eq_index >= 0:
            try: self.model.eq_active[self.eq_index] = 1
            except Exception: pass
        mj.mj_forward(self.model, self.data)
        print("[latch] End‑effector latched to panel grasp point (pose fixed).")


def _clip_vec(v: np.ndarray, lim: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= lim or lim <= 0: return v
    return (lim / (n + 1e-12)) * v


class VirtualContact:
    """
    6‑DoF spring‑damper “pose hold” between two bodies. Applies equal & opposite
    wrenches to keep the relative transform constant.
    """
    def __init__(self, model, data, body1="EE_ur10", body2="wam_grasp_point",
                 kp_pos=5000.0, kd_pos=120.0, kp_rot=300.0, kd_rot=15.0,
                 f_lim=1200.0, m_lim=300.0):
        self.model = model; self.data = data
        self.b1 = id_or_fail(model, mjtObj.mjOBJ_BODY, body1)
        self.b2 = id_or_fail(model, mjtObj.mjOBJ_BODY, body2)
        self.enabled = False
        self.kp_pos = float(kp_pos); self.kd_pos = float(kd_pos)
        self.kp_rot = float(kp_rot); self.kd_rot = float(kd_rot)
        self.f_lim  = float(f_lim);  self.m_lim  = float(m_lim)
        self.p12_des = np.zeros(3); self.q12_des = np.array([1,0,0,0], float)
        self.last_F_b2 = np.zeros(3); self.last_M_b2 = np.zeros(3)
        self.last_pos_err = 0.0; self.last_rot_err = 0.0

    def set_rot_gains(self, kp_rot: float, kd_rot: float):
        self.kp_rot = float(kp_rot); self.kd_rot = float(kd_rot)

    def set_limits(self, f_lim: float, m_lim: float):
        self.f_lim = float(f_lim); self.m_lim = float(m_lim)

    def enable_hold_current(self):
        p1 = np.array(self.data.xpos[self.b1], float)
        p2 = np.array(self.data.xpos[self.b2], float)
        q1 = normalize_quat(np.array(self.data.xquat[self.b1], float))
        q2 = normalize_quat(np.array(self.data.xquat[self.b2], float))
        R1 = quat_to_mat(q1)
        self.p12_des = R1.T @ (p2 - p1)
        self.q12_des = normalize_quat(quat_mul(quat_conj(q1), q2))
        self.enabled = True
        print("[contact] Pose hold enabled at current relative transform.")

    def disable(self):
        self.enabled = False
        self.data.xfrc_applied[self.b1,:] = 0.0
        self.data.xfrc_applied[self.b2,:] = 0.0

    def step(self):
        if not self.enabled:
            return
        p1  = np.array(self.data.xpos[self.b1], float)
        p2  = np.array(self.data.xpos[self.b2], float)
        q1  = normalize_quat(np.array(self.data.xquat[self.b1], float))
        q2  = normalize_quat(np.array(self.data.xquat[self.b2], float))
        R1  = quat_to_mat(q1)

        v1, w1 = world_body_vel(self.model, self.data, self.b1)
        v2, w2 = world_body_vel(self.model, self.data, self.b2)

        p_attach_on_b1 = p1 + R1 @ self.p12_des
        q2_des = normalize_quat(quat_mul(q1, self.q12_des))

        e_p = p2 - p_attach_on_b1
        v_attach_on_b1 = v1 + np.cross(w1, R1 @ self.p12_des)
        e_v = v2 - v_attach_on_b1
        F = -self.kp_pos * e_p - self.kd_pos * e_v

        e_r = quat_error_small(q2_des, q2)
        e_w = w2 - w1
        Mrot = -self.kp_rot * e_r - self.kd_rot * e_w

        F    = _clip_vec(F,   self.f_lim)
        Mrot = _clip_vec(Mrot, self.m_lim)

        pcom1 = np.array(self.data.xipos[self.b1], float)
        pcom2 = np.array(self.data.xipos[self.b2], float)
        r1 = (p_attach_on_b1 - pcom1)
        r2 = (p2 - pcom2)

        tau2 = Mrot + np.cross(r2, F)
        tau1 = -Mrot + np.cross(r1, -F)

        self.data.xfrc_applied[self.b1, 0:3] = -F
        self.data.xfrc_applied[self.b1, 3:6] = tau1
        self.data.xfrc_applied[self.b2, 0:3] =  F
        self.data.xfrc_applied[self.b2, 3:6] = tau2

        self.last_F_b2 = F.copy()
        self.last_M_b2 = tau2.copy()
        self.last_pos_err = float(np.linalg.norm(e_p))
        self.last_rot_err = float(np.linalg.norm(e_r))


# =============================================================================
# ------------------------------ WORLD POSE HOLD -------------------------------
# =============================================================================

class WorldPoseHold:
    """Applies a body‑frame wrench to keep a given body rigid in world."""
    def __init__(self, model, data, body_name: str,
                 kp_pos=BASE_VH_POS_KP, kd_pos=BASE_VH_POS_KD,
                 kp_rot=BASE_VH_ROT_KP, kd_rot=BASE_VH_ROT_KD):
        self.model = model; self.data = data
        self.bid = mj.mj_name2id(model, mjtObj.mjOBJ_BODY, body_name)
        if self.bid < 0:
            for nm in ["fix_x", "hinge_link"]:
                self.bid = mj.mj_name2id(model, mjtObj.mjOBJ_BODY, nm)
                if self.bid >= 0:
                    body_name = nm; break
        if self.bid < 0:
            raise RuntimeError(f"[base-hold] Body '{body_name}' not found.")
        self.kp_pos = float(kp_pos); self.kd_pos = float(kd_pos)
        self.kp_rot = float(kp_rot); self.kd_rot = float(kd_rot)
        self.enabled = False
        self.p_des = np.zeros(3); self.q_des = np.array([1,0,0,0], float)
        self.body_name = body_name

    def capture_current_as_target(self):
        self.p_des = np.array(self.data.xpos[self.bid], float)
        self.q_des = normalize_quat(np.array(self.data.xquat[self.bid], float))
        self.enabled = True
        print(f"[base-hold] Captured world pose of '{self.body_name}' as target.")

    def disable(self):
        self.enabled = False
        self.data.xfrc_applied[self.bid,:] = 0.0

    def step(self):
        if not self.enabled: return
        p  = np.array(self.data.xpos[self.bid], float)
        q  = normalize_quat(np.array(self.data.xquat[self.bid], float))
        v, w = world_body_vel(self.model, self.data, self.bid)

        e_p = p - self.p_des
        F   = -self.kp_pos * e_p - self.kd_pos * v

        e_r = quat_error_small(self.q_des, q)
        M   = -self.kp_rot * e_r - self.kd_rot * w

        self.data.xfrc_applied[self.bid, 0:3] = F
        self.data.xfrc_applied[self.bid, 3:6] = M


# =============================================================================
# ------------------------------ POMDP LOGIC ----------------------------------
# =============================================================================

REG_NAMES = ["threshold", "saw", "hysteresis", "sinusoid", "sliding"]
N_REG = len(REG_NAMES)

def friction_torque(theta: float, dtheta: float, reg: int, p: StribeckParameters) -> float:
    """Regime‑dependent frictional torque."""
    if reg == 0:  # threshold (static near zero velocity)
        return p.Fs * np.sign(dtheta) * (abs(dtheta) < 0.02)
    if reg == 1:  # saw‑tooth in angle (asymmetric)
        phase = (theta % 0.5) / 0.5
        return p.Fc * (phase - 0.5)
    if reg == 2:  # hysteresis: sign memory
        return p.Fc * np.tanh(5.0 * dtheta) + 0.30 * p.Fc * np.sign(theta)
    if reg == 3:  # sinusoid vs angle
        return p.Fc * np.sin(2.0 * np.pi * theta / 1.0)
    if reg == 4:  # sliding vs velocity
        return p.Fc * np.tanh(20.0 * dtheta)
    raise ValueError(f"Unknown regime {reg}")

def sigmoid(x: float) -> float: return 1.0/(1.0+np.exp(-x))

def transition_matrix(a: float, dtheta: float, theta: float,
                      p: StribeckParameters, eps_any: float = 0.02) -> np.ndarray:
    """Right‑stochastic regime transition model."""
    N = N_REG; T = np.zeros((N, N), float)
    tau_break = 0.65 * p.Fs
    tau_lock  = 0.45 * p.Fs
    v_small   = 0.04
    v_slide   = 0.20
    k_tau = 6.0; k_v = 5.0
    U = np.full((N, N), 1.0/N)
    for i in range(N):
        pref = np.full(N, eps_any)
        if i == 4: stay = sigmoid(k_v * (abs(dtheta) - v_slide))
        elif i == 0: stay = sigmoid(k_tau * (tau_lock - abs(a))) * sigmoid(k_v * (v_small - abs(dtheta)))
        else: stay = 0.6 * sigmoid(k_v * (v_small - abs(dtheta))) + 0.2
        pref[i] += 0.6 * stay
        to_slide  = sigmoid(k_tau * (abs(a) - tau_break)) + 0.5 * sigmoid(k_v * (abs(dtheta) - v_slide))
        to_thresh = sigmoid(k_tau * (tau_lock      - abs(a))) * sigmoid(k_v * (v_small - abs(dtheta)))
        pref[4] += 0.5 * to_slide
        pref[0] += 0.3 * to_thresh
        if i in (1,2,3):
            for j in (1,2,3):
                if j != i: pref[j] += 0.1 * (1.0 - stay)
        pref = np.clip(pref, 1e-9, None); pref /= pref.sum()
        T[i] = pref
    T = (1.0 - eps_any) * T + eps_any * U
    T /= T.sum(axis=1, keepdims=True)
    return T


def pomdp_only_ctrl(action_levels: np.ndarray,
                    params: StribeckParameters,
                    theta_ref: List[float],
                    dt: float,
                    p_max: float,
                    tau_max: float,
                    J_est: float,
                    D_est: float,
                    w_track: float,
                    w_vel: float,
                    w_u: float,
                    w_risk: float,
                    w_du: float,
                    w_sat: float,
                    w_rail: float,
                    sat_alpha: float,
                    rail_beta: float,
                    du_max: float):
    """
    Discrete action selection with a short horizon rollout; includes Δu
    regularization (w_du) applied at the first stage cost.
    """

    J_est = float(J_est); D_est = float(D_est)
    H = int(POMDP_H_CTRL)
    gamma = float(POMDP_GAMMA)
    tau_sat = float(np.max(np.abs(action_levels)))
    rng = np.random.default_rng()
    optimizer = DeltaUSequenceOptimizer(
        DeltaUOptimizerConfig(
            horizon=H,
            tau_sat=tau_sat,
            du_max=float(du_max),
            cem_samples=int(POMDP_CEM_SAMPLES),
            cem_elites=int(POMDP_CEM_ELITES),
            cem_iters=int(POMDP_CEM_ITERS),
            cem_init_std=float(POMDP_CEM_INIT_STD),
            mppi_samples=int(POMDP_MPPI_SAMPLES),
            mppi_lambda=float(POMDP_MPPI_LAMBDA),
            mppi_sigma=float(POMDP_MPPI_SIGMA),
        ),
        rng=rng,
    )

    def sequence_cost(u_seq, theta, dtheta, belief, last_u_ref, th_ref_next, dth_ref_next):
        th = theta; dth = dtheta
        csum = 0.0

        # (5) Δu penalty encourages smoothness without steady‑state lag
        u0 = float(u_seq[0])
        csum += w_du * (u0 - last_u_ref)**2

        # (3) Immediate actuator saturation risk: function of |u| only
        r_now = 1.0 if abs(u0) > tau_max else 0.0
        csum += w_risk * r_now
        if r_now > (p_max - 0.05):
            csum += 10.0 * (r_now - (p_max - 0.05))

        # Horizon rollout with belief‑weighted friction
        for h in range(H):
            u = float(u_seq[h])
            tau_f_bel = 0.0
            for j in range(N_REG):
                tau_f_bel += belief[j] * friction_torque(th, dth, j, params)

            ddth = (u - tau_f_bel - D_est*dth) / J_est
            dth  = dth + dt*ddth
            th   = th  + dt*dth

            w = (gamma**h)
            csum += w * (
                w_track*(th - th_ref_next)**2
                + w_vel*(dth - dth_ref_next)**2
                + w_u*(u*u)
            )
            sat_excess = max(0.0, abs(u) - sat_alpha * tau_sat)
            csum += w * (w_sat * (sat_excess**2))
            csum += w * (w_rail * (1.0 if abs(u) > rail_beta * tau_sat else 0.0))

            # Gentle discounted saturation awareness during rollout
            r_step = 1.0 if abs(u) > tau_max else 0.0
            csum += w * (0.2 * w_risk) * r_step

        return csum

    def cem_optimize(theta, dtheta, belief, last_u_ref, th_ref_next, dth_ref_next):
        def cost_fn(u_seq):
            return sequence_cost(u_seq, theta, dtheta, belief, last_u_ref, th_ref_next, dth_ref_next)

        best_u0, _ = optimizer.cem_optimize(cost_fn, last_u_ref)
        return best_u0

    def mppi_optimize(theta, dtheta, belief, last_u_ref, th_ref_next, dth_ref_next):
        def cost_fn(u_seq):
            return sequence_cost(u_seq, theta, dtheta, belief, last_u_ref, th_ref_next, dth_ref_next)

        u0, _ = optimizer.mppi_optimize(cost_fn, last_u_ref)
        return u0

    def inner(k, theta, dtheta, belief, last_u_ref):
        k  = int(k)
        k1 = min(k + 1, len(theta_ref)-1)
        k0 = min(k,     len(theta_ref)-1)

        # Hold the nearest future setpoint constant over the short horizon
        th_ref_next  = float(theta_ref[k1])
        dth_ref_next = (theta_ref[k1] - theta_ref[k0]) / dt

        if POMDP_CTRL_MODE == "mppi":
            return mppi_optimize(theta, dtheta, belief, last_u_ref, th_ref_next, dth_ref_next)
        if POMDP_CTRL_MODE == "cem":
            return cem_optimize(theta, dtheta, belief, last_u_ref, th_ref_next, dth_ref_next)

        best_u = 0.0; best_c = np.inf

        for u in action_levels:
            th = theta; dth = dtheta
            csum = 0.0

            # (5) Δu penalty encourages smoothness without steady‑state lag
            csum += w_du * (u - last_u_ref)**2

            # (3) Immediate actuator saturation risk: function of |u| only
            r_now = 1.0 if abs(u) > tau_max else 0.0
            csum += w_risk * r_now
            if r_now > (p_max - 0.05):
                csum += 10.0 * (r_now - (p_max - 0.05))

            # Horizon rollout with belief‑weighted friction
            for h in range(H):
                tau_f_bel = 0.0
                for j in range(N_REG):
                    tau_f_bel += belief[j] * friction_torque(th, dth, j, params)

                ddth = (u - tau_f_bel - D_est*dth) / J_est
                dth  = dth + dt*ddth
                th   = th  + dt*dth

                w = (gamma**h)
                csum += w * (
                    w_track*(th - th_ref_next)**2
                    + w_vel*(dth - dth_ref_next)**2
                    + w_u*(u*u)
                )
                sat_excess = max(0.0, abs(u) - sat_alpha * tau_sat)
                csum += w * (w_sat * (sat_excess**2))
                csum += w * (w_rail * (1.0 if abs(u) > rail_beta * tau_sat else 0.0))

                # Gentle discounted saturation awareness during rollout
                r_step = 1.0 if abs(u) > tau_max else 0.0
                csum += w * (0.2 * w_risk) * r_step

            if csum < best_c:
                best_c = csum
                best_u = float(u)

        return best_u

    return inner


class HingePOMDP:
    """
    POMDP controller for a 1‑DoF hinge in the MuJoCo model.
    Implements: (1) correct residuals, (2) correct inertia, (3) saturation risk,
    (5) Δu penalty, (6) action densification near zero, (7) integral bias.
    """
    def __init__(self, model, data, joint_name="hinge_joint",
                 dt=CTRL_DT,
                 tau_sat=POMDP_TAU_SAT,
                 sp_speed=POMDP_SP_SPEED_RAD_S,
                 params=POMDP_FRIC,
                 J_est=POMDP_J_EST,
                 D_est=POMDP_D_EST):
        self.model = model; self.data = data
        self.jid  = id_or_fail(model, mjtObj.mjOBJ_JOINT, joint_name)
        self.qadr = model.jnt_qposadr[self.jid]
        self.vadr = model.jnt_dofadr[self.jid]
        self.dt   = float(dt)
        self.tau_sat = float(tau_sat)
        self.sp_speed = float(sp_speed)
        self.params = params

        # Dynamics estimates
        self.J_est = float(J_est)
        self.D_est = float(D_est)

        self.theta_ref: List[float] = []
        self.k = 0
        self.belief = np.ones(N_REG, float) / N_REG
        self.ctrl_fun = pomdp_only_ctrl(
            ACTIONS, self.params, self.theta_ref, self.dt,
            p_max=P_MAX_RISK, tau_max=TAU_MAX_FOR_RISK,
            J_est=self.J_est, D_est=self.D_est,
            w_track=POMDP_W_TRACK, w_vel=POMDP_W_VEL,
            w_u=POMDP_W_U, w_risk=POMDP_W_RISK,
            w_du=POMDP_W_DU, w_sat=POMDP_W_SAT, w_rail=POMDP_W_RAIL,
            sat_alpha=POMDP_SAT_ALPHA, rail_beta=POMDP_RAIL_BETA,
            du_max=POMDP_DU_MAX
        )

        self.sp      = 0.0
        self.sp_goal = 0.0
        self.enabled = False
        self.prev_v  = 0.0
        self._a_hat  = None
        self.last_tau_cmd = 0.0
        self.last_risk    = 0.0
        self.last_costs = {
            "track": 0.0,
            "vel": 0.0,
            "u": 0.0,
            "du": 0.0,
            "sat": 0.0,
            "rail": 0.0,
        }

        # (7) integral bias to kill steady‑state error
        self.bias_u = 0.0
        self.Ki_bias = 0.6               # N·m per rad·s
        self.bias_limit = 0.35*self.tau_sat

    def enable_and_capture(self):
        q = float(self.data.qpos[self.qadr])
        self.sp = self.sp_goal = q
        self.theta_ref[:] = [q, q]
        self.prev_v = float(self.data.qvel[self.vadr])
        self._a_hat = 0.0
        self.belief[:] = 1.0/N_REG
        self.k = 0
        self.enabled = True
        print(f"[hinge-POMDP] Enabled at q={q:.4f} rad; tau_sat={self.tau_sat:.1f} N·m; J_est={self.J_est:.4f}")

    def set_target_relative(self, dq: float):
        lo, hi = (-np.inf, np.inf)
        if self.model.jnt_limited[self.jid] == 1: lo, hi = self.model.jnt_range[self.jid]
        q = float(self.data.qpos[self.qadr])
        self.sp_goal = clamp(q + float(dq), lo, hi)
        print(f"[hinge-POMDP] Target REL → goal={self.sp_goal:.4f} rad")

    def set_target_absolute(self, q_abs: float):
        lo, hi = (-np.inf, np.inf)
        if self.model.jnt_limited[self.jid] == 1: lo, hi = self.model.jnt_range[self.jid]
        self.sp_goal = clamp(float(q_abs), lo, hi)
        print(f"[hinge-POMDP] Target ABS → goal={self.sp_goal:.4f} rad")

    def step(self):
        if not self.enabled: return
        dt = self.dt
        q  = float(self.data.qpos[self.qadr])
        dq = float(self.data.qvel[self.vadr])

        # ramp setpoint
        err_goal = self.sp_goal - self.sp
        max_step = self.sp_speed * dt
        if abs(err_goal) <= max_step:
            self.sp = self.sp_goal
        else:
            self.sp += math.copysign(max_step, err_goal)

        self.theta_ref.append(self.sp)

        # (7) integrate small bias to remove steady-state error
        e_sp = (self.sp - q)
        self.bias_u += self.Ki_bias * e_sp * dt
        self.bias_u = clamp(self.bias_u, -self.bias_limit, self.bias_limit)

        # choose torque using Δu penalty and add bias
        prev_u = self.last_tau_cmd
        u_raw = float(self.ctrl_fun(self.k, q, dq, self.belief, self.last_tau_cmd)) + self.bias_u
        # keep optional smoothing path but alpha=0 by default
        u = (1.0 - POMDP_U_SMOOTH_ALPHA) * u_raw + POMDP_U_SMOOTH_ALPHA * self.last_tau_cmd
        u = clamp(u, -self.tau_sat, self.tau_sat)
        self.data.qfrc_applied[self.vadr] += u
        self.last_tau_cmd = u

        # (1) observation update (residual) in **torque units**
        a_raw = (dq - self.prev_v) / dt
        self.prev_v = dq
        if self._a_hat is None:
            self._a_hat = a_raw
        else:
            self._a_hat = 0.2*a_raw + 0.8*self._a_hat
        a_meas = self._a_hat

        tau_dyn = self.J_est * a_meas + self.D_est * dq

        e = np.empty(N_REG)
        for j in range(N_REG):
            tau_f_j = friction_torque(q, dq, j, self.params)
            e[j] = (u - tau_dyn) - (tau_f_j)  # (no external FF per request)

        L = np.exp(-0.5 * (e / SIGMA_RESIDUAL)**2)
        L = np.clip(L, 1e-12, None)

        T = transition_matrix(u, dq, q, self.params)
        b_pred = self.belief @ T
        b_post = b_pred * L
        b_post = np.clip(b_post, BELIEF_FLOOR, None)
        b_post /= b_post.sum()
        self.belief = b_post

        # (3) risk = actuator saturation probability surrogate
        self.last_risk = float(1.0 if abs(u) > TAU_MAX_FOR_RISK else 0.0)

        dth_ref = (self.theta_ref[-1] - self.theta_ref[-2]) / dt if len(self.theta_ref) > 1 else 0.0
        sat_excess = max(0.0, abs(u) - POMDP_SAT_ALPHA * self.tau_sat)
        self.last_costs = {
            "track": POMDP_W_TRACK * (q - self.sp)**2,
            "vel": POMDP_W_VEL * (dq - dth_ref)**2,
            "u": POMDP_W_U * (u * u),
            "du": POMDP_W_DU * (u - prev_u)**2,
            "sat": POMDP_W_SAT * (sat_excess**2),
            "rail": POMDP_W_RAIL * (1.0 if abs(u) > POMDP_RAIL_BETA * self.tau_sat else 0.0),
        }

        self.k += 1


class HingeLoad:
    """Sinusoidal resistance torque applied at the hinge."""
    def __init__(self, model, data, joint_name="hinge_joint",
                 amp_frac=LOAD_SINE_AMP_FRAC, freq_hz=LOAD_SINE_FREQ_HZ, phase=LOAD_PHASE,
                 tau_sat=POMDP_TAU_SAT):
        self.model = model; self.data = data
        self.jid  = id_or_fail(model, mjtObj.mjOBJ_JOINT, joint_name)
        self.vadr = model.jnt_dofadr[self.jid]
        self.amp  = float(amp_frac) * float(tau_sat)
        self.freq = float(freq_hz); self.phase = float(phase)
        self.enabled = False; self.t0 = 0.0; self.last_tau = 0.0

    def enable(self, t_now: float):
        self.enabled = True; self.t0 = float(t_now)
        print(f"[load] Oscillatory resistance enabled (amp={self.amp:.1f} N·m @ {self.freq:.2f} Hz).")

    def step(self, t_now: float):
        if not self.enabled:
            self.last_tau = 0.0; return
        tau = self.amp * math.sin(2*math.pi*self.freq*(t_now - self.t0) + self.phase)
        self.data.qfrc_applied[self.vadr] += tau
        self.last_tau = tau


# =============================================================================
# ------------------------------- STATE MACHINE --------------------------------
# =============================================================================

class Phase(Enum):
    PREGRASP = auto()
    APPROACH = auto()
    ALIGN    = auto()
    SETTLE   = auto()
    DRIVE    = auto()
    HOLD     = auto()

class GraspSM:
    """
    0) Pre‑grasp
    1) Approach
    2) Align    → latch (pose hold); unlock UR to passive; close fingertips
    3) Settle   → release hinge freeze; enable POMDP; start sinusoidal load
    4) Drive
    """
    def __init__(self, model, data, ur: UR10, hand: BarrettHand,
                 freezer_panel: JointFreezer, freezer_wam: JointGroupFreezer,
                 latch: LatchPair, v_contact: VirtualContact,
                 hinge_pomdp: HingePOMDP, hinge_load: HingeLoad,
                 base_freeze: JointGroupFreezer, base_hold: WorldPoseHold,
                 plot_queue: "queue.Queue[dict]"):
        self.model = model; self.data = data
        self.ur = ur; self.hand = hand
        self.freezer_panel = freezer_panel
        self.freezer_wam   = freezer_wam
        self.latch         = latch
        self.v_contact     = v_contact
        self.hinge         = hinge_pomdp
        self.hload         = hinge_load
        self.base_freeze   = base_freeze
        self.base_hold     = base_hold
        self.plot_q        = plot_queue

        self.panel_bid = self._first_body(["solar_frame","hinge_link"])
        self.edge_bid  = self._first_body(EDGE_BODY_PREFERENCE)
        self.phase = Phase.PREGRASP
        self.t_phase2_start = None
        self.t_latched = None
        self._t_last_plot = -1e9

    def _first_body(self, names: List[str]) -> int:
        for nm in names:
            i = mj.mj_name2id(self.model, mjtObj.mjOBJ_BODY, nm)
            if i >= 0: return i
        raise RuntimeError(f"None of bodies found: {names}")

    def panel_axes_edge(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        R = np.array(self.data.xmat[self.panel_bid], float).reshape(3,3)
        n_panel = R[:,2]
        edge_dir = R[:,0] if EDGE_AXIS.lower().startswith("x") else R[:,1]
        p_edge = np.array(self.data.xpos[self.edge_bid], float)
        p_edge = p_edge + np.array([0.0, 0.0, -VERT_OFFSET_BELOW_M])  # world −Z offset
        return n_panel, edge_dir, p_edge

    def step(self, sim_time: float, dt: float):
        # Base rigid in world
        if not self.base_freeze.enabled:
            self.base_freeze.enable(self.data)
        self.base_freeze.apply(self.data)
        self.base_hold.step()

        # Equality row follows until latch
        self.latch.follow_tick()

        # Hold hinge until drive begins; WAM freeze after 3 s
        self.freezer_panel.apply(self.data)
        if (not self.freezer_wam.enabled) and (sim_time >= WAM_FREEZE_DELAY_S):
            self.freezer_wam.enable(self.data)
        self.freezer_wam.apply(self.data)

        n_panel, edge_dir, p_edge = self.panel_axes_edge()
        desired_world_dir = edge_dir

        if self.phase == Phase.PREGRASP:
            self.hand.open()
            p_des = p_edge - PREGRASP_DIST*n_panel
            self.ur.ik_to_point(p_des)
            if np.linalg.norm(np.array(self.data.site_xpos[self.ur.site_id]) - p_des) < 0.012:
                self.phase = Phase.APPROACH

        elif self.phase == Phase.APPROACH:
            p_des = p_edge - APPROACH_DIST*n_panel
            self.ur.ik_to_point(p_des)
            if np.linalg.norm(np.array(self.data.site_xpos[self.ur.site_id]) - p_des) < 0.007:
                self.ur.enable_joint6_only()
                self.t_phase2_start = sim_time
                self.phase = Phase.ALIGN

        elif self.phase == Phase.ALIGN:
            err = self.ur.yaw_align_step_joint6(desired_world_dir)
            timed_out = (self.t_phase2_start is not None and
                         (sim_time - self.t_phase2_start) > YAW_ALIGN_TIMEOUT_S)
            if (err is not None and abs(err) < YAW_TOL) or timed_out:
                self.latch.latch_now()
                self.v_contact.enable_hold_current()

                self.hand.close_partial_no_rotation(TIP_CLOSE_FRAC)

                if UR_PASSIVE_AFTER_LATCH:
                    self.ur.set_passive(True)
                    if abs(UR_DAMP_SCALE_ON_PASSIVE - 1.0) > 1e-9:
                        self.ur.scale_damping(UR_DAMP_SCALE_ON_PASSIVE)
                    print("[UR] Unlocked: passive follow enabled.")

                self.t_latched = sim_time
                self.phase = Phase.SETTLE

        elif self.phase == Phase.SETTLE:
            if self.t_latched is not None and (sim_time - self.t_latched) >= POST_LATCH_SETTLE_S:
                self.freezer_panel.release()
                self.hinge.enable_and_capture()
                self.hinge.set_target_relative(math.radians(40.0))
                if not self.hload.enabled:
                    self.hload.enable(sim_time)
                print("[phase] Hinge drive active (POMDP).")
                self.phase = Phase.DRIVE

        elif self.phase == Phase.DRIVE:
            self.v_contact.step()
            self.hinge.step()
            self.hload.step(sim_time)

        # UR/hand torques
        self.ur.step_pd()
        self.hand.step_pd()

        # telemetry → plot queue
        self._maybe_post_plot(sim_time)

    def _maybe_post_plot(self, t: float):
        if (t - self._t_last_plot) < (1.0 / PLOT_RATE_HZ): return
        self._t_last_plot = t

        angle = float(self.data.qpos[self.hinge.qadr]) if self.hinge.enabled else 0.0
        setpoint = float(self.hinge.sp) if self.hinge.enabled else 0.0
        tau_cmd  = float(self.hinge.last_tau_cmd) if self.hinge.enabled else 0.0
        tau_load = float(self.hload.last_tau) if self.hload.enabled else 0.0

        F = self.v_contact.last_F_b2 if self.v_contact.enabled else np.zeros(3)
        M = self.v_contact.last_M_b2 if self.v_contact.enabled else np.zeros(3)
        pos_err = self.v_contact.last_pos_err if self.v_contact.enabled else 0.0
        rot_err = self.v_contact.last_rot_err if self.v_contact.enabled else 0.0
        risk    = self.hinge.last_risk if self.hinge.enabled else 0.0
        costs   = self.hinge.last_costs if self.hinge.enabled else {}

        pkt = dict(t=t, angle=angle, setpoint=setpoint,
                   tau_cmd=tau_cmd, tau_load=tau_load,
                   F=F, M=M, pos_err=pos_err, rot_err=rot_err, risk=risk)
        pkt.update({
            "c_track": costs.get("track", 0.0),
            "c_vel": costs.get("vel", 0.0),
            "c_u": costs.get("u", 0.0),
            "c_du": costs.get("du", 0.0),
            "c_sat": costs.get("sat", 0.0),
            "c_rail": costs.get("rail", 0.0),
        })
        try:
            self.plot_q.put_nowait(pkt)
        except queue.Full:
            try: _ = self.plot_q.get_nowait()
            except queue.Empty: pass
            try: self.plot_q.put_nowait(pkt)
            except queue.Full: pass


# =============================================================================
# -------------------------------- PLOTTING UI --------------------------------
# =============================================================================

class RollingBuffer:
    """Convenience rolling buffer for 1D streams."""
    def __init__(self, maxlen: int = 6000):
        self.maxlen = int(maxlen)
        self.x = []

    def append(self, v):
        self.x.append(v)
        if len(self.x) > self.maxlen:
            del self.x[:len(self.x)-self.maxlen]

    def data(self):
        return self.x


class TelemetryRecorder:
    """Optional disk logging."""
    def __init__(self, save_csv: bool = SAVE_CSV, save_npz: bool = SAVE_NPZ, out_dir: str = SAVE_DIR):
        self.save_csv = bool(save_csv)
        self.save_npz = bool(save_npz)
        self.dir = out_dir
        self.rows: List[Dict[str, float]] = []
        self._csv_header_written = False
        if (self.save_csv or self.save_npz) and not os.path.isdir(self.dir):
            os.makedirs(self.dir, exist_ok=True)

    def add(self, pkt: Dict[str, float]):
        if self.save_csv or self.save_npz:
            self.rows.append({
                "t": pkt["t"],
                "angle": pkt["angle"],
                "setpoint": pkt["setpoint"],
                "tau_cmd": pkt["tau_cmd"],
                "tau_load": pkt["tau_load"],
                "Fx": float(pkt["F"][0]),
                "Fy": float(pkt["F"][1]),
                "Fz": float(pkt["F"][2]),
                "Mx": float(pkt["M"][0]),
                "My": float(pkt["M"][1]),
                "Mz": float(pkt["M"][2]),
                "pos_err": pkt["pos_err"],
                "rot_err": pkt["rot_err"],
                "risk": pkt["risk"],
                "c_track": pkt["c_track"],
                "c_vel": pkt["c_vel"],
                "c_u": pkt["c_u"],
                "c_du": pkt["c_du"],
                "c_sat": pkt["c_sat"],
                "c_rail": pkt["c_rail"],
            })

    def flush(self):
        if not (self.save_csv or self.save_npz) or not self.rows:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if self.save_csv:
            path = os.path.join(self.dir, f"telemetry_{stamp}.csv")
            cols = list(self.rows[0].keys())
            with open(path, "w", encoding="utf-8") as f:
                f.write(",".join(cols) + "\n")
                for r in self.rows:
                    f.write(",".join(str(r[c]) for c in cols) + "\n")
            print(f"[log] CSV written: {path}")
        if self.save_npz:
            path = os.path.join(self.dir, f"telemetry_{stamp}.npz")
            arr = {k: np.array([r[k] for r in self.rows], float) for k in self.rows[0].keys()}
            np.savez(path, **arr)
            print(f"[log] NPZ written: {path}")
        arr = {k: np.array([r[k] for r in self.rows], float) for k in self.rows[0].keys()}
        abs_u = np.abs(arr["tau_cmd"])
        mean_abs_u = float(abs_u.mean()) if abs_u.size else 0.0
        max_abs_u = float(abs_u.max()) if abs_u.size else 0.0
        frac_08 = float((abs_u > (0.8 * POMDP_TAU_SAT)).mean()) if abs_u.size else 0.0
        frac_095 = float((abs_u > (0.95 * POMDP_TAU_SAT)).mean()) if abs_u.size else 0.0
        cost_totals = {
            "track": float(arr["c_track"].sum()),
            "vel": float(arr["c_vel"].sum()),
            "u": float(arr["c_u"].sum()),
            "du": float(arr["c_du"].sum()),
            "sat": float(arr["c_sat"].sum()),
            "rail": float(arr["c_rail"].sum()),
        }
        print(
            "[log] Torque stats: mean|u|={:.3f}, max|u|={:.3f}, frac>|0.8|={:.3f}, frac>|0.95|={:.3f}".format(
                mean_abs_u, max_abs_u, frac_08, frac_095
            )
        )
        print(
            "[log] Cost breakdown: track={track:.3f}, vel={vel:.3f}, u={u:.3f}, du={du:.3f}, "
            "sat={sat:.3f}, rail={rail:.3f}".format(**cost_totals)
        )
        self.rows.clear()


class LivePlotterUI:
    """Live plotting UI (main thread)."""
    _fields = ("t angle setpoint tau_cmd tau_load F M pos_err rot_err risk").split()

    def __init__(self, rate_hz=PLOT_RATE_HZ, queue_size=PLOT_QUEUE_SIZE,
                 window_sec=PLOT_WINDOW_SEC):
        self.queue: "queue.Queue[dict]" = queue.Queue(maxsize=queue_size)
        self.rate  = float(rate_hz)
        self.window_sec = float(window_sec)
        self._stop_flag = False
        self.rec = TelemetryRecorder()

    def stop(self): self._stop_flag = True

    def start_blocking(self):
        import matplotlib
        try:
            import matplotlib.pyplot as plt
        except Exception:
            matplotlib.use("TkAgg", force=True)
            import matplotlib.pyplot as plt

        plt.ion()
        fig, axs = plt.subplots(3, 2, figsize=(12, 8))
        ((axA, axB), (axC, axD), (axE, axF)) = axs

        la, = axA.plot([], [], label="hinge angle (rad)")
        lr, = axA.plot([], [], label="setpoint (rad)")
        axA.set_title("Panel hinge angle"); axA.set_xlabel("time (s)"); axA.set_ylabel("angle (rad)"); axA.legend(loc="upper left")

        lct, = axB.plot([], [], label="controller torque (N·m)")
        llt, = axB.plot([], [], label="oscillatory load torque (N·m)")
        axB.set_title("Resistance torques at hinge"); axB.set_xlabel("time (s)"); axB.set_ylabel("torque (N·m)"); axB.legend(loc="upper left")

        lfx, = axC.plot([], [], label="Fx"); lfy, = axC.plot([], [], label="Fy"); lfz, = axC.plot([], [], label="Fz")
        axC.set_title("Force applied to panel (world)"); axC.set_xlabel("time (s)"); axC.set_ylabel("force (N)"); axC.legend(loc="upper left")

        lmx, = axD.plot([], [], label="Mx"); lmy, = axD.plot([], [], label="My"); lmz, = axD.plot([], [], label="Mz")
        axD.set_title("Resistance torque at contact (world)"); axD.set_xlabel("time (s)"); axD.set_ylabel("torque (N·m)"); axD.legend(loc="upper left")

        lpe, = axE.plot([], [], label="|pos error| (m)")
        lre, = axE.plot([], [], label="|rot error| (rad)")
        axE.set_title("Relative pose error (contact)"); axE.set_xlabel("time (s)"); axE.set_ylabel("error"); axE.legend(loc="upper left")

        axF.axis("off")
        txt = axF.text(0.02, 0.95, "Live diagnostics", va="top", ha="left")

        t  = RollingBuffer()
        a  = RollingBuffer(); r = RollingBuffer()
        ct = RollingBuffer(); lt = RollingBuffer()
        Fx = RollingBuffer(); Fy = RollingBuffer(); Fz = RollingBuffer()
        Mx = RollingBuffer(); My = RollingBuffer(); Mz = RollingBuffer()
        epos = RollingBuffer(); erot = RollingBuffer(); risk = RollingBuffer()

        def on_timer():
            drained = False
            while True:
                try:
                    pkt = self.queue.get_nowait()
                    drained = True
                    self.rec.add(pkt)

                    t.append(pkt["t"])
                    a.append(pkt["angle"]); r.append(pkt["setpoint"])
                    ct.append(pkt["tau_cmd"]); lt.append(pkt["tau_load"])
                    Fx.append(float(pkt["F"][0])); Fy.append(float(pkt["F"][1])); Fz.append(float(pkt["F"][2]))
                    Mx.append(float(pkt["M"][0])); My.append(float(pkt["M"][1])); Mz.append(float(pkt["M"][2]))
                    epos.append(pkt["pos_err"]); erot.append(pkt["rot_err"]); risk.append(pkt["risk"])
                except queue.Empty:
                    break

            if drained:
                la.set_data(t.data(), a.data())
                lr.set_data(t.data(), r.data())
                lct.set_data(t.data(), ct.data()); llt.set_data(t.data(), lt.data())
                lfx.set_data(t.data(), Fx.data()); lfy.set_data(t.data(), Fy.data()); lfz.set_data(t.data(), Fz.data())
                lmx.set_data(t.data(), Mx.data()); lmy.set_data(t.data(), My.data()); lmz.set_data(t.data(), Mz.data())
                lpe.set_data(t.data(), epos.data()); lre.set_data(t.data(), erot.data())

                for ax in (axA, axB, axC, axD, axE):
                    ax.relim(); ax.autoscale_view()

                last_risk = risk.data()[-1] if risk.data() else 0.0
                txt.set_text(f"Live diagnostics\nQueueing @ {self.rate:.1f} Hz\nrisk={last_risk:.2f}")

                fig.canvas.draw_idle()

            if self._stop_flag and self.queue.empty():
                try:
                    self.rec.flush()
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception:
                    pass

        timer = fig.canvas.new_timer(interval=int(1000.0/self.rate))
        timer.add_callback(on_timer)
        timer.start()

        import matplotlib.pyplot as plt
        plt.show(block=True)


# =============================================================================
# ------------------------------- HELPERS -------------------------------------
# =============================================================================

def try_disable_equality(model, data, eq_name: str):
    """Best‑effort: disable a named equality at load if runtime supports eq_active."""
    try:
        eq_id = mj.mj_name2id(model, mjtObj.mjOBJ_EQUALITY, eq_name)
    except Exception:
        eq_id = -1
    if eq_id >= 0 and hasattr(model, "eq_active"):
        try:
            model.eq_active[eq_id] = 0
            mj.mj_forward(model, data)
            print(f"[eq] Disabled equality '{eq_name}' at load.")
        except Exception:
            print(f"[eq] Could not disable equality '{eq_name}' (ignored).")
    else:
        print(f"[eq] Equality '{eq_name}' not disabled (not found or no eq_active).")


# =============================================================================
# ------------------------------- SIM THREAD ----------------------------------
# =============================================================================

def simulation_thread(model, data, total_time_s, realtime, plotter_ui: LivePlotterUI):
    """
    MuJoCo simulation loop in a worker thread.
    """
    ur        = UR10(model, data)
    hand      = BarrettHand(model, data)
    latch     = LatchPair(model, data, body1="EE_ur10", body2="wam_grasp_point")
    contact   = VirtualContact(model, data, body1="EE_ur10", body2="wam_grasp_point",
                               kp_pos=5000.0, kd_pos=120.0, kp_rot=300.0, kd_rot=15.0,
                               f_lim=1200.0, m_lim=300.0)
    freezer_panel = JointFreezer(model, data, FREEZE_PANEL_JOINT)
    freezer_wam   = JointGroupFreezer(model, data, FREEZE_WAM_JOINTS)

    base_freeze = JointGroupFreezer(model, data, BASE_JOINTS_TO_FREEZE)
    base_hold   = WorldPoseHold(model, data, BASE_BODY_NAME,
                                kp_pos=BASE_VH_POS_KP, kd_pos=BASE_VH_POS_KD,
                                kp_rot=BASE_VH_ROT_KP, kd_rot=BASE_VH_ROT_KD)
    base_hold.capture_current_as_target()

    # (2) Compute hinge equivalent inertia from the model
    try:
        J_eq = hinge_equivalent_inertia(model, data, FREEZE_PANEL_JOINT)
    except Exception:
        J_eq = POMDP_J_EST
    print(f"[hinge] Equivalent inertia estimated: J_eq={J_eq:.4f} kg·m^2")

    hinge_ctrl = HingePOMDP(model, data, joint_name=FREEZE_PANEL_JOINT,
                            dt=CTRL_DT, tau_sat=POMDP_TAU_SAT,
                            sp_speed=POMDP_SP_SPEED_RAD_S, params=POMDP_FRIC,
                            J_est=J_eq, D_est=POMDP_D_EST)
    hinge_load = HingeLoad(model, data, joint_name=FREEZE_PANEL_JOINT,
                           amp_frac=LOAD_SINE_AMP_FRAC, freq_hz=LOAD_SINE_FREQ_HZ,
                           phase=LOAD_PHASE, tau_sat=POMDP_TAU_SAT)

    sm   = GraspSM(model, data, ur, hand, freezer_panel, freezer_wam,
                   latch, contact, hinge_ctrl, hinge_load,
                   base_freeze, base_hold, plotter_ui.queue)

    # Optional viewer in the sim thread
    have_viewer = False
    viewer = None
    try:
        import mujoco.viewer as mjv
        viewer = mjv.launch_passive(model, data); have_viewer = True
        print("Viewer launched.")
    except Exception:
        pass

    h = model.opt.timestep
    substeps = max(1, int(round(CTRL_DT / h)))
    nsteps   = int(round(total_time_s / CTRL_DT))
    t0 = time.perf_counter()

    for k in range(nsteps):
        sim_time = (k+1)*CTRL_DT
        sm.step(sim_time, CTRL_DT)
        for _ in range(substeps):
            mj.mj_step(model, data)
        if have_viewer:
            try: viewer.sync()
            except Exception: have_viewer = False
        if realtime:
            now = time.perf_counter() - t0
            tgt = (k+1)*CTRL_DT
            if now < tgt: time.sleep(tgt - now)

    if have_viewer:
        try: viewer.close()
        except Exception: pass

    plotter_ui.stop()
    print("Simulation thread finished.")


# =============================================================================
# ---------------------------------- MAIN -------------------------------------
# =============================================================================

def main(xml_path="wamhingev4.xml", realtime=True, total_time_s=TOTAL_TIME_S):
    print(f"Loading: {xml_path}")
    model = mj.MjModel.from_xml_path(xml_path)
    data  = mj.MjData(model)
    mj.mj_forward(model, data)

    # Optional seed UR10 initial pose (from XML numerics, if useful)
    try:
        jn = ["ur10_joint_1","ur10_joint_2","ur10_joint_3","ur10_joint_4","ur10_joint_5","ur10_joint_6"]
        jids = [id_or_fail(model, mjtObj.mjOBJ_JOINT, n) for n in jn]
        qadr = [model.jnt_qposadr[j] for j in jids]
        q0 = [-1.75924,-0.75396,-2.57644,-1.2566,-1.57075,0.0]
        for a,v in zip(qadr,q0): data.qpos[a] = v
        mj.mj_forward(model, data)
    except Exception:
        pass

    if DISABLE_EQ_FIXER_AT_LOAD:
        try_disable_equality(model, data, FIXER_EQ_NAME)

    # Start UI (main thread) and simulation (worker thread)
    plotter = LivePlotterUI(rate_hz=PLOT_RATE_HZ, queue_size=PLOT_QUEUE_SIZE)
    sim_thr = threading.Thread(target=simulation_thread,
                               args=(model, data, total_time_s, realtime, plotter),
                               daemon=True)
    sim_thr.start()
    try:
        plotter.start_blocking()
    finally:
        plotter.stop()
        sim_thr.join(timeout=5.0)
    print("Done.")


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    xml = "wamhingev4.xml"; rt = True; T = TOTAL_TIME_S
    if len(sys.argv) > 1: xml = sys.argv[1]
    if len(sys.argv) > 2: rt  = bool(int(sys.argv[2]))
    if len(sys.argv) > 3: T   = float(sys.argv[3])
    main(xml, rt, T)
6
