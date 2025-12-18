#!/usr/bin/env python3
"""
UR10e attached to panel; satellite base rigid in world; **hinge PID position control**.

- Base is held still (freeze + world-pose hold).
- UR10 goes passive after attachment and follows the panel motion.
- Hand-to-panel **attachment** maintained by a 6‑DoF force coupling
  (position/orientation regulation via wrenches).
- WAM locked after 3 s; pregrasp/approach with 3 cm vertical offset; joint‑6 yaw
  alignment (cap 66°, 10 s timeout); optional tip-only close at 0.77.

NEW (live plotting):
- Async plot process with **one figure and subplots**, updated at **10 Hz**.
- Queues data from the sim loop; non-blocking (drops samples if overloaded).
- Subplots show:
  (1) Hinge angle & setpoint
  (2) Resistance torque at hinge (controller output) + **oscillatory load torque**
  (3) **Force applied to panel** (Fx,Fy,Fz)
  (4) Resistance torque at contact (Mx,My,Mz)
  (5) Relative pose error (‖pos‖, ‖rot‖)

NEW (this revision):
- Adds an **oscillatory load torque** at the hinge (sinusoidal), with a short period
  to make the PID work hard. Tuned via HINGE_DIST_* constants below.
"""

import sys, time, math, collections, atexit
from typing import List, Tuple, Optional
import numpy as np
import mujoco as mj
from mujoco import mjtObj

# ===================== timing & alignment knobs =====================
TOTAL_TIME_S         = 25.0
CTRL_DT              = 1/200

EDGE_BODY_PREFERENCE = ["flagp_wam", "flagp", "flagp_ur5e", "hinge_link"]

PREGRASP_DIST        = 0.14
APPROACH_DIST        = 0.02
VERT_OFFSET_BELOW_M  = 0.03    # 3 cm below the edge point (world −Z)

EDGE_AXIS            = "x"                   # 'x' or 'y' (panel edge to align to)
PAIR_AXIS_LOCAL      = "x"                   # 'x' or 'y' (palm axis of the 2‑finger pair)
YAW_KP               = 1.4
YAW_KD               = 0.40
YAW_MAX_STEP         = math.radians(6.0)     # per‑tick limit (~6°)
YAW_TOL              = math.radians(1.0)
YAW_TOTAL_CAP        = math.radians(66.0)    # total allowed rotation since entry
YAW_ALIGN_TIMEOUT_S  = 10.0
YAW_DIR_SIGN         = -1                    # opposite direction

FREEZE_PANEL_JOINT   = "hinge_joint"         # hold panel until attachment settles
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
POST_WELD_SETTLE_S   = 0.6

# ===================== base hold =====================
BASE_JOINTS_TO_FREEZE = ["fix_roll","fix_pitch","fix_yaw","fix_z","fix_y","fix_x"]
BASE_BODY_NAME       = "EE_sate"
BASE_VH_POS_KP       = 8000.0
BASE_VH_POS_KD       = 220.0
BASE_VH_ROT_KP       = 450.0
BASE_VH_ROT_KD       = 22.0

DISABLE_EQ_FIXER_AT_LOAD = True
FIXER_EQ_NAME            = "fixer"

# ===================== orientation lock (force coupling) =====================
ORIENT_LOCK_ENABLE  = True
VW_ROT_KP_LOCK      = 1200.0      # N*m/rad  (stiff)
VW_ROT_KD_LOCK      = 40.0        # N*m*s/rad
VW_FORCE_LIMIT_N    = 1200.0      # clamp |F| (N)
VW_TORQUE_LIMIT_NM  = 300.0       # clamp |M| (N*m)

# Force coupling base gains (pre‑lock)
VW_POS_KP           = 5000.0      # N/m
VW_POS_KD           = 120.0       # N*s/m
VW_ROT_KP           = 300.0       # N*m/rad
VW_ROT_KD           = 15.0        # N*m*s/rad

# ===================== UR passive settings =====================
UR_PASSIVE_AFTER_LATCH        = True
UR_DAMP_SCALE_ON_PASSIVE      = 1.0   # set <1.0 for freer motion (e.g., 0.4)

# ===================== hinge PID (panel angle control) =====================
HINGE_TARGET_MODE        = "relative"    # "relative" or "absolute"
HINGE_TARGET_DELTA_RAD   = math.radians(40.0)     # if mode == "relative"
HINGE_TARGET_ABS_RAD     = math.radians(0.75*180/180)

HINGE_SP_SPEED_RAD_S     = math.radians(12.0)     # setpoint slew (rad/s)

HINGE_KP                 = 200.0     # N*m / rad
HINGE_KI                 = 10.0      # N*m / (rad*s)
HINGE_KD                 = 6.0      # N*m / (rad/s)

HINGE_TORQUE_LIMIT_NM    = 80.0
HINGE_I_LIMIT_FRACTION   = 0.5
HINGE_VISC_DAMP          = 0.0
HINGE_STATIC_FF_NM       = 0.0

# ===================== Oscillatory hinge load (this revision) =====================
# Short period (e.g., 0.4 s) to challenge the PID: freq 2.5 Hz by default.
HINGE_DIST_ENABLE     = True
HINGE_DIST_AMP_NM     = 5.0            # amplitude of external load torque (N·m)
HINGE_DIST_FREQ_HZ    = 0.5             # frequency (Hz) -> period 0.4 s
HINGE_DIST_PHASE_RAD  = 0.0             # initial phase
HINGE_DIST_START_AT_PHASE4 = True       # start when hinge PID becomes active

# ===================== (disabled) EE push controller (kept for completeness) =====================
USE_EE_PUSH           = False
UR_PUSH_FORCE_N       = 45.0
UR_PUSH_SIGN          = +1

# Barrett hand – tip-only close (no rotation on the two proximal joints)
PALM_BODY_NAME       = "wam/bhand/bhand_palm_link"
TIP_CLOSE_FRAC       = 0.77

# ===================== plotting (async process; single window) =====================
PLOT_ENABLE          = True
PLOT_WINDOW_S        = 20.0
PLOT_UPDATE_HZ       = 10.0        # <= requested rate
PLOT_QUEUE_MAX       = 4096

def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x
def id_or_fail(model, objtype, name):
    i = mj.mj_name2id(model, objtype, name)
    if i < 0: raise RuntimeError(f"{name!r} not found.")
    return i

# ---------------- quat & small math ----------------
def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_conj(q):
    w,x,y,z = q
    return np.array([w, -x, -y, -z], dtype=float)

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

def normalize_quat(q):
    n = np.linalg.norm(q)
    return q if n == 0 else q / n

def quat_error_small(q_des, q_cur):
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

# ---------------- PD helper ----------------
class JointPD:
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

    def set_q_des(self, qd: np.ndarray): self.q_des = np.asarray(qd, float)
    def hold_here(self): self.q_des = np.array([self.data.qpos[a] for a in self.qadr], float)
    def set_enabled(self, on: bool): self.enabled = bool(on)

    def step(self):
        if not self.enabled:
            for aid in self.aids: self.data.ctrl[aid] = 0.0
            return
        q  = np.array([self.data.qpos[a] for a in self.qadr], float)
        dq = np.array([self.data.qvel[a] for a in self.vadr], float)
        tau = self.kp*(self.q_des - q) - self.kd*dq
        tau = np.clip(tau, -self.tau_lim, self.tau_lim)
        for i, aid in enumerate(self.aids): self.data.ctrl[aid] = float(tau[i])

# ---------------- UR10 controller ----------------
class UR10:
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

# ---------------- Barrett hand (optional close) ----------------
class BarrettHand:
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

# ---------------- qpos/qvel freezers ----------------
class JointFreezer:
    def __init__(self, model, data, joint_name: str):
        self.model = model; self.data = data
        jid = mj.mj_name2id(model, mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            print(f"[freeze] WARN: joint '{joint_name}' not found; disabled.")
            self.enabled = False; self.qadr = 0; self.vadr = 0; self.nv = 0; self.qref = np.zeros(0)
            return
        self.enabled = True
        self.qadr = model.jnt_qposadr[jid]; self.vadr = model.jnt_dofadr[jid]
        jtype = model.jnt_type[jid]
        self.nv = 6 if jtype == mj.mjtJoint.mjJNT_FREE else 3 if jtype == mj.mjtJoint.mjJNT_BALL else 1
        self.qref = np.array(data.qpos[self.qadr:self.qadr+self.nv], float)
        print(f"[freeze] Holding '{joint_name}' at qpos={self.qref.tolist()} (nv={self.nv})")
    def release(self): self.enabled = False
    def apply(self, data):
        if not self.enabled: return
        data.qpos[self.qadr:self.qadr+self.nv] = self.qref
        data.qvel[self.vadr:self.vadr+self.nv] = 0.0

class JointGroupFreezer:
    def __init__(self, model, data, joint_names: List[str]):
        self.model = model; self.data = data
        self.items = []
        for name in joint_names:
            jid = mj.mj_name2id(model, mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                print(f"[group-freeze] WARN: joint '{name}' not found; skipping."); continue
            qadr = model.jnt_qposadr[jid]; vadr = model.jnt_dofadr[jid]
            jtype = model.jnt_type[jid]
            nv = 6 if jtype == mj.mjtJoint.mjJNT_FREE else 3 if jtype == mj.mjtJoint.mjJNT_BALL else 1
            self.items.append((qadr, vadr, nv, name))
        self.enabled = False; self.qrefs = None
    def enable(self, data):
        self.qrefs = [np.array(data.qpos[qadr:qadr+nv], float) for (qadr, _, nv, _) in self.items]
        self.enabled = True
        print(f"[group-freeze] ENABLED at current pose for {len(self.items)} joints: "
              + ", ".join([nm for *_3, nm in self.items]))
    def apply(self, data):
        if not self.enabled or not self.items: return
        for (qadr, vadr, nv, _), qref in zip(self.items, self.qrefs):
            data.qpos[qadr:qadr+nv] = qref; data.qvel[vadr:vadr+nv] = 0.0

# ---------------- equality helpers ----------------
EQ_STRIDE = 7  # number of eq_data floats for any equality slot

def _find_weld_between(model, body1_name: str, body2_name: str) -> int:
    # Keep function name for compatibility; treat as general constraint row search
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

class WeldLatch:
    """Keeps constraint row following until latch, then freezes it (if present)."""
    def __init__(self, model, data, body1="EE_ur10", body2="wam_grasp_point"):
        self.model = model; self.data = data
        self.body1 = body1; self.body2 = body2
        self.eq_index = _find_weld_between(model, body1, body2)
        if self.eq_index < 0:
            print("[attach] NOTE: No constraint row found in XML for bodies; relying on force coupling.")
        self.body1_id = id_or_fail(model, mjtObj.mjOBJ_BODY, body1)
        self.body2_id = id_or_fail(model, mjtObj.mjOBJ_BODY, body2)
        self.has_eq_active = hasattr(model, "eq_active")
        self.follow_mode = True
        if self.has_eq_active and self.eq_index >= 0:
            try: model.eq_active[self.eq_index] = 0
            except Exception: pass
            mj.mj_forward(model, data)

    def follow_tick(self):
        if not self.follow_mode or self.eq_index < 0: return
        relpose = compute_relpose_in_body1_frame(self.data, self.body1_id, self.body2_id)
        _set_eq_data_row(self.model, self.eq_index, relpose)

    def latch_now(self):
        if self.eq_index >= 0:
            relpose = compute_relpose_in_body1_frame(self.data, self.body1_id, self.body2_id)
            _set_eq_data_row(self.model, self.eq_index, relpose)
        self.follow_mode = False
        if self.has_eq_active and self.eq_index >= 0:
            try: self.model.eq_active[self.eq_index] = 1
            except Exception: pass
        mj.mj_forward(self.model, self.data)
        print(f"[attach] End effector coupled to panel (constraint + force coupling).")

# ---------------- Force coupling (6-DoF wrench, world frame) ----------------
def _clip_vec(v: np.ndarray, lim: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= lim or lim <= 0: return v
    return (lim / (n + 1e-12)) * v

class VirtualWeld:
    """6‑DoF force coupling between EE and panel grasp point."""
    def __init__(self, model, data, body1="EE_ur10", body2="wam_grasp_point",
                 kp_pos=VW_POS_KP, kd_pos=VW_POS_KD, kp_rot=VW_ROT_KP, kd_rot=VW_ROT_KD,
                 f_lim=VW_FORCE_LIMIT_N, m_lim=VW_TORQUE_LIMIT_NM):
        self.model = model; self.data = data
        self.b1 = id_or_fail(model, mjtObj.mjOBJ_BODY, body1)  # EE/hand
        self.b2 = id_or_fail(model, mjtObj.mjOBJ_BODY, body2)  # panel grasp point
        self.enabled = False
        self.kp_pos = float(kp_pos); self.kd_pos = float(kd_pos)
        self.kp_rot = float(kp_rot); self.kd_rot = float(kd_rot)
        self.f_lim  = float(f_lim);  self.m_lim  = float(m_lim)
        self.p12_des = np.zeros(3); self.q12_des = np.array([1,0,0,0], float)
        # captured for plotting
        self.last_force_hand   = np.zeros(3)
        self.last_torque_hand  = np.zeros(3)
        self.last_force_panel  = np.zeros(3)
        self.last_torque_panel = np.zeros(3)
        self.last_pos_err = 0.0
        self.last_rot_err = 0.0

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
        print(f"[attach] Force coupling enabled with fixed relative pose.")

    def disable(self):
        self.enabled = False
        self.data.xfrc_applied[self.b1,:] = 0.0
        self.data.xfrc_applied[self.b2,:] = 0.0

    def step(self):
        if not self.enabled:
            return
        # World poses
        p1  = np.array(self.data.xpos[self.b1], float)
        p2  = np.array(self.data.xpos[self.b2], float)
        q1  = normalize_quat(np.array(self.data.xquat[self.b1], float))
        q2  = normalize_quat(np.array(self.data.xquat[self.b2], float))
        R1  = quat_to_mat(q1)

        # World velocities (at COM)
        v1, w1 = world_body_vel(self.model, self.data, self.b1)
        v2, w2 = world_body_vel(self.model, self.data, self.b2)

        # Desired child world pose on b1
        p_attach_on_b1 = p1 + R1 @ self.p12_des
        q2_des = normalize_quat(quat_mul(q1, self.q12_des))

        # Translational spring-damper
        e_p = p2 - p_attach_on_b1
        v_attach_on_b1 = v1 + np.cross(w1, R1 @ self.p12_des)
        e_v = v2 - v_attach_on_b1
        F = -self.kp_pos * e_p - self.kd_pos * e_v   # world force

        # Rotational spring-damper (orientation hold)
        e_r = quat_error_small(q2_des, q2)  # small-angle axis error (world)
        e_w = w2 - w1
        Mrot = -self.kp_rot * e_r - self.kd_rot * e_w

        # Clamp for stability
        F    = _clip_vec(F,   self.f_lim)
        Mrot = _clip_vec(Mrot, self.m_lim)

        # Apply equal & opposite wrenches at COMs
        pcom1 = np.array(self.data.xipos[self.b1], float)
        pcom2 = np.array(self.data.xipos[self.b2], float)
        r1 = (p_attach_on_b1 - pcom1)
        r2 = (p2 - pcom2)

        tau2 = Mrot + np.cross(r2, F)   # on panel side
        tau1 = -Mrot + np.cross(r1, -F) # on EE/hand side

        self.data.xfrc_applied[self.b1, 0:3] = -F
        self.data.xfrc_applied[self.b1, 3:6] = tau1
        self.data.xfrc_applied[self.b2, 0:3] =  F
        self.data.xfrc_applied[self.b2, 3:6] = tau2

        # capture for plotting
        self.last_force_hand   = (-F).copy()
        self.last_torque_hand  = tau1.copy()
        self.last_force_panel  = F.copy()
        self.last_torque_panel = tau2.copy()
        self.last_pos_err = float(np.linalg.norm(e_p))
        self.last_rot_err = float(np.linalg.norm(e_r))

# ---------------- World-pose hold (applies wrench only to the base) ----------------
class WorldPoseHold:
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
            raise RuntimeError("WorldPoseHold: base body not found; check BASE_BODY_NAME.")
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

# ---------------- Hinge PID (torque output, ramped setpoint) ----------------
class HingePID:
    def __init__(self, model, data, joint_name="hinge_joint",
                 kp=HINGE_KP, ki=HINGE_KI, kd=HINGE_KD,
                 sp_speed=HINGE_SP_SPEED_RAD_S,
                 tau_lim=HINGE_TORQUE_LIMIT_NM,
                 i_frac=HINGE_I_LIMIT_FRACTION,
                 visc=HINGE_VISC_DAMP,
                 tau_static=HINGE_STATIC_FF_NM):
        self.model = model; self.data = data
        self.jid  = id_or_fail(model, mjtObj.mjOBJ_JOINT, joint_name)
        self.qadr = model.jnt_qposadr[self.jid]
        self.vadr = model.jnt_dofadr[self.jid]      # single-DoF index
        self.kp = float(kp); self.ki = float(ki); self.kd = float(kd)
        self.sp_speed = float(sp_speed)
        self.tau_lim  = float(tau_lim)
        self.i_lim_nm = float(abs(i_frac) * self.tau_lim)
        self.visc     = float(visc)
        self.tau_static = float(tau_static)
        self.enabled = False
        self.sp      = 0.0      # current setpoint
        self.spdot   = 0.0      # setpoint rate used for D-on-error
        self.sp_goal = 0.0
        self.i_accum = 0.0
        self.limited = (self.model.jnt_limited[self.jid] == 1)
        self.lo, self.hi = (self.model.jnt_range[self.jid] if self.limited else (-np.inf, np.inf))
        self.last_tau = 0.0  # controller torque for plotting

    def _clamp_to_range(self, val):
        return clamp(val, self.lo, self.hi) if self.limited else val

    def enable_and_capture(self):
        q = float(self.data.qpos[self.qadr])
        self.sp = self.sp_goal = q
        self.spdot = 0.0
        self.i_accum = 0.0
        self.enabled = True
        print(f"[hinge-PID] Enabled at q={q:.4f} rad.")

    def set_target_absolute(self, q_abs: float):
        self.sp_goal = self._clamp_to_range(float(q_abs))
        print(f"[hinge-PID] Target ABS set to {self.sp_goal:.4f} rad (clamped).")

    def set_target_relative(self, dq: float):
        q = float(self.data.qpos[self.qadr])
        self.sp_goal = self._clamp_to_range(q + float(dq))
        print(f"[hinge-PID] Target REL set: q_now={q:.4f} -> goal={self.sp_goal:.4f} rad (clamped).")

    def step(self, dt: float):
        if not self.enabled: return
        # Ramped setpoint
        q = float(self.data.qpos[self.qadr])
        dq = float(self.data.qvel[self.vadr])

        err_goal = self.sp_goal - self.sp
        max_step = self.sp_speed * dt
        if abs(err_goal) <= max_step:
            self.sp = self.sp_goal
            self.spdot = 0.0
        else:
            self.sp += math.copysign(max_step, err_goal)
            self.spdot = math.copysign(self.sp_speed, err_goal)

        # PID with D on error (includes setpoint rate)
        e     = self.sp - q
        edot  = self.spdot - dq

        # Integrator with anti-windup on output
        if self.ki > 0.0:
            i_max = self.i_lim_nm / max(self.ki, 1e-9)
            self.i_accum = clamp(self.i_accum + e*dt, -i_max, i_max)
            tau_i = self.ki * self.i_accum
        else:
            tau_i = 0.0

        tau_p = self.kp * e
        tau_d = self.kd * edot
        tau_v = -self.visc * dq
        tau_ff = math.copysign(self.tau_static, e) if abs(e) > 1e-5 else 0.0

        tau = tau_p + tau_i + tau_d + tau_v + tau_ff
        tau = clamp(tau, -self.tau_lim, self.tau_lim)

        # IMPORTANT: set (not accumulate) so it doesn't grow across steps.
        self.data.qfrc_applied[self.vadr] = tau
        self.last_tau = float(tau)

# ---------------- Oscillatory hinge load (sinusoidal) ----------------
class HingeDisturbance:
    """
    Adds a time-varying **oscillatory load torque** at the hinge:
        tau_dist(t) = AMP * sin(2π f t + phase)
    This is applied in addition to the controller torque.
    """
    def __init__(self, model, data, hinge_pid: HingePID,
                 amp_nm=HINGE_DIST_AMP_NM, freq_hz=HINGE_DIST_FREQ_HZ, phase=HINGE_DIST_PHASE_RAD):
        self.model = model; self.data = data
        self.pid = hinge_pid
        self.amp = float(amp_nm)
        self.freq = float(freq_hz)
        self.phase = float(phase)
        self.enabled = bool(HINGE_DIST_ENABLE)
        self.last_tau = 0.0

    def set_enabled(self, on: bool):
        self.enabled = bool(on)

    def step(self, t_now: float):
        if not self.enabled:
            self.last_tau = 0.0
            return
        tau = self.amp * math.sin(2.0*math.pi*self.freq*t_now + self.phase)
        # Add on top of whatever the PID wrote this tick
        self.data.qfrc_applied[self.pid.vadr] += tau
        self.last_tau = float(tau)

# ---------------- (optional) EE push ----------------
class EENormalForceController:
    def __init__(self, model, data, ee_body_name="EE_ur10", site_name="ur10_attachment_site",
                 force_N=UR_PUSH_FORCE_N, sign=UR_PUSH_SIGN):
        self.model = model; self.data = data
        self.ee_bid   = id_or_fail(model, mjtObj.mjOBJ_BODY, ee_body_name)
        self.site_id  = id_or_fail(model, mjtObj.mjOBJ_SITE, site_name)
        self.force_N  = float(force_N)
        self.sign     = int(np.sign(sign) if sign != 0 else 1)
        self.enabled  = False
    def enable(self):  self.enabled = True
    def disable(self): self.enabled = False; self.clear()
    def clear(self):
        self.data.xfrc_applied[self.ee_bid, :] = 0.0
    def step(self, n_panel_world: np.ndarray):
        self.clear()
        if not self.enabled: return
        n = np.asarray(n_panel_world, float); n = n / (np.linalg.norm(n) + 1e-12)
        F = (self.sign * self.force_N) * n
        p_apply = np.array(self.data.site_xpos[self.site_id], float)
        p_com   = np.array(self.data.xipos[self.ee_bid], float)
        r = p_apply - p_com
        tau = np.cross(r, F)
        self.data.xfrc_applied[self.ee_bid, 0:3] = F
        self.data.xfrc_applied[self.ee_bid, 3:6] = tau

# ---------------- state machine ----------------
class GraspSM:
    """
    0) Pre‑grasp: IK to (p_edge - PREGRASP_DIST*n_panel) + vertical offset
    1) Approach : IK to (p_edge - APPROACH_DIST*n_panel) + vertical offset → enable joint6‑only yaw
    2) Yaw align (joint6 only). When |err|<YAW_TOL or timeout:
         - Attach: constraint row (if present) fixed; force‑coupling enabled
         - Optional: close fingers to 0.77 (no rotation)
         - Unlock UR (passive) so it follows the panel
         - Boost rotational gains (orientation hold)
    3) Short settle; then:
         - release hinge freeze
         - enable **hinge PID** with target (relative/absolute) and ramped setpoint
         - (this revision) enable **oscillatory load torque** at hinge
    4) Hold: base rigid in world; coupling runs; hinge PID drives angle; load torque present
    """
    def __init__(self, model, data, ur: UR10, hand: BarrettHand,
                 freezer_panel: JointFreezer, freezer_wam: JointGroupFreezer,
                 latch: WeldLatch, v_weld: VirtualWeld,
                 hinge_pid: HingePID, hinge_dist: HingeDisturbance,
                 ee_force: EENormalForceController,
                 base_freeze: JointGroupFreezer, base_hold: WorldPoseHold):
        self.model = model; self.data = data
        self.ur = ur; self.hand = hand
        self.freezer_panel = freezer_panel
        self.freezer_wam   = freezer_wam
        self.latch         = latch
        self.v_weld        = v_weld
        self.hinge_pid     = hinge_pid
        self.hinge_dist    = hinge_dist
        self.ee_force      = ee_force
        self.base_freeze   = base_freeze
        self.base_hold     = base_hold

        self.panel_bid = self._first_body(["solar_frame","hinge_link"])
        self.edge_bid  = self._first_body(EDGE_BODY_PREFERENCE)
        self.phase = 0
        self.t_phase2_start = None
        self.t_latched = None

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
        # Always: keep base rigid (freeze + world-pose anchor)
        if not self.base_freeze.enabled:
            self.base_freeze.enable(self.data)
        self.base_freeze.apply(self.data)
        self.base_hold.step()

        # Keep equality row following until attach
        self.latch.follow_tick()

        # Hold hinge until attach completes; lock WAM after 3 s
        self.freezer_panel.apply(self.data)
        if (not self.freezer_wam.enabled) and (sim_time >= WAM_FREEZE_DELAY_S):
            self.freezer_wam.enable(self.data)
        self.freezer_wam.apply(self.data)

        # No EE push by default
        self.ee_force.clear()

        n_panel, edge_dir, p_edge = self.panel_axes_edge()
        desired_world_dir = edge_dir

        if self.phase == 0:
            self.hand.open()
            p_des = p_edge - PREGRASP_DIST*n_panel
            self.ur.ik_to_point(p_des)
            if np.linalg.norm(np.array(self.data.site_xpos[self.ur.site_id]) - p_des) < 0.012:
                self.phase = 1

        elif self.phase == 1:
            p_des = p_edge - APPROACH_DIST*n_panel
            self.ur.ik_to_point(p_des)
            if np.linalg.norm(np.array(self.data.site_xpos[self.ur.site_id]) - p_des) < 0.007:
                self.ur.enable_joint6_only()
                self.t_phase2_start = sim_time
                self.phase = 2

        elif self.phase == 2:
            err = self.ur.yaw_align_step_joint6(desired_world_dir)
            timed_out = (self.t_phase2_start is not None and
                         (sim_time - self.t_phase2_start) > YAW_ALIGN_TIMEOUT_S)
            if (err is not None and abs(err) < YAW_TOL) or timed_out:
                # Attach + regulate relative pose
                self.latch.latch_now()
                self.v_weld.enable_hold_current()

                # Tips close (optional)
                self.hand.close_partial_no_rotation(TIP_CLOSE_FRAC)

                # Immediately unlock UR to passive follow
                if UR_PASSIVE_AFTER_LATCH:
                    self.ur.set_passive(True)
                    if abs(UR_DAMP_SCALE_ON_PASSIVE - 1.0) > 1e-9:
                        self.ur.scale_damping(UR_DAMP_SCALE_ON_PASSIVE)
                    print("[UR] Unlocked: passive follow enabled.")

                # Orientation hold via stronger coupling gains
                if ORIENT_LOCK_ENABLE:
                    self.v_weld.set_rot_gains(VW_ROT_KP_LOCK, VW_ROT_KD_LOCK)
                    self.v_weld.set_limits(VW_FORCE_LIMIT_N, VW_TORQUE_LIMIT_NM)
                    print(f"[attach] Orientation hold: kp_rot={VW_ROT_KP_LOCK}, kd_rot={VW_ROT_KD_LOCK}")

                self.t_latched = sim_time
                self.phase = 3

        elif self.phase == 3:
            # After a short settle, release hinge freeze and enable hinge PID (+oscillatory load)
            if self.t_latched is not None and (sim_time - self.t_latched) >= POST_WELD_SETTLE_S:
                self.freezer_panel.release()

                # Enable hinge PID and set goal
                self.hinge_pid.enable_and_capture()
                if HINGE_TARGET_MODE.lower().startswith("rel"):
                    self.hinge_pid.set_target_relative(HINGE_TARGET_DELTA_RAD)
                else:
                    self.hinge_pid.set_target_absolute(HINGE_TARGET_ABS_RAD)

                # Start oscillatory disturbance at the hinge (if configured)
                if HINGE_DIST_START_AT_PHASE4:
                    self.hinge_dist.set_enabled(True)

                if USE_EE_PUSH: self.ee_force.enable()
                print("[phase] Hinge PID enabled; oscillatory load active; UR passive; coupling active.")
                self.phase = 4

        elif self.phase == 4:
            # Maintain relative pose + hinge PID + oscillatory load
            self.v_weld.step()
            self.hinge_pid.step(dt)
            self.hinge_dist.step(sim_time)
            if USE_EE_PUSH: self.ee_force.step(n_panel)

        # apply UR/hand PD torques (UR is passive now if unlocked)
        self.ur.step_pd()
        self.hand.step_pd()

# ---------------- helpers ----------------
def try_disable_equality(model, data, eq_name: str):
    """Best-effort: disable named equality via eq_active if present."""
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

# ---------------- plotting process (async, single figure, 10 Hz) ----------------
def _plotter_process_main(queue, window_s: float, update_hz: float):
    """
    Child process entrypoint. Owns a single Matplotlib window with subplots.
    Expects messages:
      (t, hinge_q, hinge_sp, tau_ctrl, tau_dist, Fx, Fy, Fz, Mx, My, Mz, pos_err, rot_err)
    Sentinel to terminate: ('__STOP__',)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as _np
        import time as _time
        import collections as _collections
    except Exception as e:
        print(f"[plotter] Matplotlib not available: {e}")
        return

    plt.ion()

    # Buffers
    maxlen = 200000
    t_buf   = _collections.deque(maxlen=maxlen)
    hq_buf  = _collections.deque(maxlen=maxlen)
    hsp_buf = _collections.deque(maxlen=maxlen)
    tc_buf  = _collections.deque(maxlen=maxlen)
    td_buf  = _collections.deque(maxlen=maxlen)
    fx_buf  = _collections.deque(maxlen=maxlen)
    fy_buf  = _collections.deque(maxlen=maxlen)
    fz_buf  = _collections.deque(maxlen=maxlen)
    mx_buf  = _collections.deque(maxlen=maxlen)
    my_buf  = _collections.deque(maxlen=maxlen)
    mz_buf  = _collections.deque(maxlen=maxlen)
    pe_buf  = _collections.deque(maxlen=maxlen)
    re_buf  = _collections.deque(maxlen=maxlen)

    # One figure with 3x2 subplots (use 5)
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    ax1, ax2 = axs[0,0], axs[0,1]
    ax3, ax4 = axs[1,0], axs[1,1]
    ax5      = axs[2,0]
    ax6      = axs[2,1]  # free for notes

    # Lines
    (line_hq,)  = ax1.plot([], [], label="hinge angle (rad)")
    (line_hsp,) = ax1.plot([], [], label="setpoint (rad)")
    ax1.set_title("Panel hinge angle"); ax1.set_xlabel("time (s)"); ax1.set_ylabel("angle (rad)")
    ax1.legend(loc="best")

    (line_tc,) = ax2.plot([], [], label="controller torque (N·m)")
    (line_td,) = ax2.plot([], [], label="oscillatory load torque (N·m)")
    ax2.set_title("Resistance torques at hinge"); ax2.set_xlabel("time (s)"); ax2.set_ylabel("torque (N·m)")
    ax2.legend(loc="best")

    (line_fx,) = ax3.plot([], [], label="Fx")
    (line_fy,) = ax3.plot([], [], label="Fy")
    (line_fz,) = ax3.plot([], [], label="Fz")
    ax3.set_title("Force applied to panel (world)"); ax3.set_xlabel("time (s)"); ax3.set_ylabel("force (N)")
    ax3.legend(loc="best")

    (line_mx,) = ax4.plot([], [], label="Mx")
    (line_my,) = ax4.plot([], [], label="My")
    (line_mz,) = ax4.plot([], [], label="Mz")
    ax4.set_title("Resistance torque at contact (world)"); ax4.set_xlabel("time (s)"); ax4.set_ylabel("torque (N·m)")
    ax4.legend(loc="best")

    (line_pe,) = ax5.plot([], [], label="‖pos error‖ (m)")
    (line_re,) = ax5.plot([], [], label="‖rot error‖ (rad)")
    ax5.set_title("Relative pose error (contact)"); ax5.set_xlabel("time (s)"); ax5.set_ylabel("error")
    ax5.legend(loc="best")

    ax6.axis("off")
    ax6.text(0.02, 0.8, "Live diagnostics", fontsize=11)
    ax6.text(0.02, 0.6, f"Queueing @ {update_hz:.1f} Hz", fontsize=10)

    fig.tight_layout()
    try:
        plt.show(block=False)
    except Exception:
        pass

    t0 = None
    update_dt = 1.0 / max(1e-6, update_hz)
    next_draw = _time.perf_counter()
    running = True

    def drain_queue():
        nonlocal t0, running
        while True:
            try:
                msg = queue.get_nowait()
            except Exception:
                break
            if isinstance(msg, tuple) and len(msg) == 1 and msg[0] == '__STOP__':
                running = False; break
            try:
                (t, hq, hsp, tc, td, fx, fy, fz, mx, my, mz, pe, re) = msg
            except Exception:
                continue
            if t0 is None: t0 = float(t)
            tt = float(t) - t0
            t_buf.append(tt);  hq_buf.append(float(hq));  hsp_buf.append(float(hsp))
            tc_buf.append(float(tc)); td_buf.append(float(td))
            fx_buf.append(float(fx)); fy_buf.append(float(fy)); fz_buf.append(float(fz))
            mx_buf.append(float(mx)); my_buf.append(float(my)); mz_buf.append(float(mz))
            pe_buf.append(float(pe)); re_buf.append(float(re))

    def select_window(arr_t, arr_y, win):
        if len(arr_t) == 0:
            return _np.array([]), _np.array([])
        t_arr = _np.fromiter(arr_t, dtype=float)
        y_arr = _np.fromiter(arr_y, dtype=float)
        tmax = t_arr[-1]; tmin = max(0.0, tmax - win)
        idx0 = _np.searchsorted(t_arr, tmin, side="left")
        return t_arr[idx0:], y_arr[idx0:]

    while running:
        drain_queue()
        try:
            if not plt.get_fignums():
                break
        except Exception:
            break

        now = _time.perf_counter()
        if now >= next_draw:
            # 1) hinge
            x, y1 = select_window(t_buf, hq_buf, window_s)
            _, y2 = select_window(t_buf, hsp_buf, window_s)
            line_hq.set_data(x, y1); line_hsp.set_data(x, y2)
            xlim = (max(0.0, (x[-1] if x.size else 0.0) - window_s), (x[-1] if x.size else window_s))
            ax1.set_xlim(*xlim); ax1.relim(); ax1.autoscale_view(scaley=True)

            # 2) torques
            x, tc = select_window(t_buf, tc_buf, window_s)
            _, td = select_window(t_buf, td_buf, window_s)
            line_tc.set_data(x, tc); line_td.set_data(x, td)
            xlim = (max(0.0, (x[-1] if x.size else 0.0) - window_s), (x[-1] if x.size else window_s))
            ax2.set_xlim(*xlim); ax2.relim(); ax2.autoscale_view(scaley=True)

            # 3) forces
            x, fx = select_window(t_buf, fx_buf, window_s)
            _, fy = select_window(t_buf, fy_buf, window_s)
            _, fz = select_window(t_buf, fz_buf, window_s)
            line_fx.set_data(x, fx); line_fy.set_data(x, fy); line_fz.set_data(x, fz)
            xlim = (max(0.0, (x[-1] if x.size else 0.0) - window_s), (x[-1] if x.size else window_s))
            ax3.set_xlim(*xlim); ax3.relim(); ax3.autoscale_view(scaley=True)

            # 4) moments
            x, mx = select_window(t_buf, mx_buf, window_s)
            _, my = select_window(t_buf, my_buf, window_s)
            _, mz = select_window(t_buf, mz_buf, window_s)
            line_mx.set_data(x, mx); line_my.set_data(x, my); line_mz.set_data(x, mz)
            xlim = (max(0.0, (x[-1] if x.size else 0.0) - window_s), (x[-1] if x.size else window_s))
            ax4.set_xlim(*xlim); ax4.relim(); ax4.autoscale_view(scaley=True)

            # 5) errors
            x, pe = select_window(t_buf, pe_buf, window_s)
            _, re = select_window(t_buf, re_buf, window_s)
            line_pe.set_data(x, pe); line_re.set_data(x, re)
            xlim = (max(0.0, (x[-1] if x.size else 0.0) - window_s), (x[-1] if x.size else window_s))
            ax5.set_xlim(*xlim); ax5.relim(); ax5.autoscale_view(scaley=True)

            try:
                fig.canvas.draw_idle()
                plt.pause(0.001)
            except Exception:
                pass
            next_draw = now + update_dt
        else:
            try:
                plt.pause(0.001)
            except Exception:
                pass

    try:
        plt.close('all')
    except Exception:
        pass

class AsyncPlotter:
    """
    Parent-side facade for the plotting process.
    push(...) is non-blocking; drops samples if queue is full to avoid interfering with sim.
    """
    def __init__(self, window_s: float, update_hz: float, qmax: int = PLOT_QUEUE_MAX):
        try:
            import multiprocessing as mp
        except Exception as e:
            print(f"[plot] multiprocessing unavailable: {e}")
            self.enabled = False; return
        self.enabled = True
        import multiprocessing as mp
        self.mp = mp
        self.queue = self.mp.Queue(maxsize=int(qmax))
        self.proc  = self.mp.Process(target=_plotter_process_main,
                                     args=(self.queue, float(window_s), float(update_hz)),
                                     daemon=True)
        self.proc.start()
        atexit.register(self.close)

    def push(self, t, hinge_q, hinge_sp, tau_ctrl, tau_dist,
             Fx, Fy, Fz, Mx, My, Mz, pos_err, rot_err):
        if not self.enabled: return
        msg = (float(t), float(hinge_q), float(hinge_sp), float(tau_ctrl), float(tau_dist),
               float(Fx), float(Fy), float(Fz), float(Mx), float(My), float(Mz),
               float(pos_err), float(rot_err))
        try:
            self.queue.put_nowait(msg)
        except Exception:
            pass  # queue full -> drop this sample

    def close(self):
        if not self.enabled: return
        try:
            self.queue.put_nowait(('__STOP__',))
        except Exception:
            pass
        try:
            self.proc.join(timeout=1.5)
        except Exception:
            pass
        self.enabled = False

# ---------------- main ----------------
def main(xml_path="wamhingev4.xml", realtime=True, total_time_s=TOTAL_TIME_S):
    print(f"Loading: {xml_path}")
    model = mj.MjModel.from_xml_path(xml_path)
    data  = mj.MjData(model)
    mj.mj_forward(model, data)

    # Optional: seed UR10 initial pose (from XML numerics)
    try:
        jn = ["ur10_joint_1","ur10_joint_2","ur10_joint_3","ur10_joint_4","ur10_joint_5","ur10_joint_6"]
        jids = [id_or_fail(model, mjtObj.mjOBJ_JOINT, n) for n in jn]
        qadr = [model.jnt_qposadr[j] for j in jids]
        q0 = [-1.75924,-0.75396,-2.57644,-1.2566,-1.57075,0.0]
        for a,v in zip(qadr,q0): data.qpos[a] = v
        mj.mj_forward(model, data)
    except Exception:
        pass

    # Try to disable pre-existing equality ('fixer') if the runtime supports it
    if DISABLE_EQ_FIXER_AT_LOAD:
        try_disable_equality(model, data, FIXER_EQ_NAME)

    ur        = UR10(model, data)
    hand      = BarrettHand(model, data)

    latch     = WeldLatch(model, data, body1="EE_ur10", body2="wam_grasp_point")
    v_weld    = VirtualWeld(model, data, body1="EE_ur10", body2="wam_grasp_point",
                            kp_pos=VW_POS_KP, kd_pos=VW_POS_KD, kp_rot=VW_ROT_KP, kd_rot=VW_ROT_KD,
                            f_lim=VW_FORCE_LIMIT_N, m_lim=VW_TORQUE_LIMIT_NM)

    freezer_panel = JointFreezer(model, data, FREEZE_PANEL_JOINT)
    freezer_wam   = JointGroupFreezer(model, data, FREEZE_WAM_JOINTS)

    # Base rigid hold (freeze + world-pose anchor)
    base_freeze = JointGroupFreezer(model, data, BASE_JOINTS_TO_FREEZE)
    base_hold   = WorldPoseHold(model, data, BASE_BODY_NAME,
                                kp_pos=BASE_VH_POS_KP, kd_pos=BASE_VH_POS_KD,
                                kp_rot=BASE_VH_ROT_KP, kd_rot=BASE_VH_ROT_KD)
    base_hold.capture_current_as_target()

    hinge_pid = HingePID(model, data, joint_name=FREEZE_PANEL_JOINT,
                         kp=HINGE_KP, ki=HINGE_KI, kd=HINGE_KD,
                         sp_speed=HINGE_SP_SPEED_RAD_S,
                         tau_lim=HINGE_TORQUE_LIMIT_NM,
                         i_frac=HINGE_I_LIMIT_FRACTION,
                         visc=HINGE_VISC_DAMP,
                         tau_static=HINGE_STATIC_FF_NM)

    hinge_dist = HingeDisturbance(model, data, hinge_pid,
                                  amp_nm=HINGE_DIST_AMP_NM,
                                  freq_hz=HINGE_DIST_FREQ_HZ,
                                  phase=HINGE_DIST_PHASE_RAD)
    # Optionally defer enablement to when PID becomes active.
    hinge_dist.set_enabled(HINGE_DIST_ENABLE and not HINGE_DIST_START_AT_PHASE4)

    ee_force  = EENormalForceController(model, data, ee_body_name="EE_ur10",
                                        site_name="ur10_attachment_site",
                                        force_N=UR_PUSH_FORCE_N, sign=UR_PUSH_SIGN)

    sm   = GraspSM(model, data, ur, hand, freezer_panel, freezer_wam, latch, v_weld,
                   hinge_pid, hinge_dist, ee_force, base_freeze, base_hold)

    # Timing & viewer
    h = model.opt.timestep
    substeps = max(1, int(round(CTRL_DT / h)))
    nsteps   = int(round(total_time_s / CTRL_DT))

    have_viewer = False
    try:
        import mujoco.viewer as mjv
        viewer = mjv.launch_passive(model, data); have_viewer = True
        print("Viewer launched.")
    except Exception:
        viewer = None

    # --- async live plotter ---
    plotter = None
    if PLOT_ENABLE:
        try:
            plotter = AsyncPlotter(PLOT_WINDOW_S, PLOT_UPDATE_HZ, qmax=PLOT_QUEUE_MAX)
        except Exception as e:
            print(f"[plot] disabled: {e}")
            plotter = None

    t0 = time.perf_counter()
    try:
        for k in range(nsteps):
            sim_time = (k+1)*CTRL_DT
            sm.step(sim_time, CTRL_DT)

            for _ in range(substeps):
                mj.mj_step(model, data)

            # push signals for plotting (non-blocking)
            if plotter is not None:
                q_hinge  = float(data.qpos[hinge_pid.qadr])
                sp_now   = float(hinge_pid.sp) if hinge_pid.enabled else q_hinge
                tau_ctrl = float(hinge_pid.last_tau)          # controller torque
                tau_dist = float(hinge_dist.last_tau)         # oscillatory load torque
                # Use **force applied to panel** and torque at panel contact:
                Fx, Fy, Fz = v_weld.last_force_panel.tolist()
                Mx, My, Mz = v_weld.last_torque_panel.tolist()
                pe = float(v_weld.last_pos_err)
                re = float(v_weld.last_rot_err)
                plotter.push(sim_time, q_hinge, sp_now, tau_ctrl, tau_dist,
                             Fx, Fy, Fz, Mx, My, Mz, pe, re)

            if have_viewer:
                try: viewer.sync()
                except Exception: have_viewer = False
            if realtime:
                now = time.perf_counter() - t0
                tgt = (k+1)*CTRL_DT
                if now < tgt: time.sleep(tgt - now)
    finally:
        if have_viewer:
            try: viewer.close()
            except Exception: pass
        if plotter is not None:
            plotter.close()
    print("Done.")

if __name__ == "__main__":
    # Helps on Windows when using multiprocessing
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass

    xml = "wamhingev4.xml"; rt = True; T = TOTAL_TIME_S
    if len(sys.argv) > 1: xml = sys.argv[1]
    if len(sys.argv) > 2: rt  = bool(int(sys.argv[2]))
    if len(sys.argv) > 3: T   = float(sys.argv[3])
    main(xml, rt, T)
