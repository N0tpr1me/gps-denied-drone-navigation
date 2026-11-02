"""Named airframes: a real vehicle's numbers for the env, derived in one place from its
source files. ``StructureHoldConfig.from_dict`` merges the mapping of ``airframe: <name>``
under whatever the config states itself, so a config can start from a vehicle and still
override any single number.

``placeholder``: the 0.8 kg 5-inch-class quad every policy up to v4 trained on (the config
defaults, nothing merged).

``gz_x500``: the vehicle PX4's Gazebo SITL flies (``sim2real/docker``, PX4 v1.17.0), the one
the module is benchmarked on against tuned PX4. Sources, copied below as constants:

- PX4-gazebo-models ``x500_base/model.sdf`` and ``x500/model.sdf``: base_link 2.0 kg with
  diagonal inertia (0.021667, 0.021667, 0.04); four rotors of 0.016077 kg at (±0.174,
  ±0.174) m; MulticopterMotorModel motorConstant 8.54858e-06 N/(rad/s)^2, momentConstant
  0.016 m, maxRotVelocity 1000 rad/s, timeConstantUp 0.0125 s, timeConstantDown 0.025 s.
- PX4 ``airframes/4001_gz_x500``: CA_ROTOR0..3 at (0.13, ±0.22) and (-0.13, ±0.20) m with KM
  ±0.05 (PX4's allocation geometry, not the SDF's); SIM_GZ_EC_MIN / MAX 150 / 1000 rad/s, so
  rotor speed is linear in the normalised command and thrust quadratic in it;
  MPC_THR_HOVER 0.60 (only the hover thrust estimator's start value).
- ``eval/baselines/px4_tuned_v1.params``: the autotuned roll and pitch rate gains every
  benchmark flies with; yaw gains are stock.
- The module's normalised thrust clamp, 0.08 to 0.9 (``sim2real/px4/structure_hold``).

What ``gz_x500()`` derives from them:

- mass and inertia: base plus rotors (point masses at the arm ends plus their own inertia).
- max thrust 4 x 8.54858e-06 x 1000^2 = 34.2 N (thrust/weight 1.69); hover at a normalised
  command of 0.729, where PX4's hover thrust estimate converges.
- the thrust curve: at hover, thrust grows 1.61x faster with the command than the straight
  line through the hover point that the module's thrust mapping assumes.
- torque per unit of PX4's normalised torque: PX4's allocation (pseudo-inverse of its own
  effectiveness matrix, roll and pitch normalised together, yaw by its largest entry)
  driving the SDF rotors through the thrust curve, linearised at hover: roll 3.41, pitch
  5.50, yaw 0.68 N m. Pitch is stronger because PX4 places the pitch arm at 0.13 m and the
  rotors sit at 0.174 m (and autotune gave pitch 25 % lower gains than roll).
- max torque, where the first rotor reaches full or zero command from hover: roll and
  pitch 2.11, yaw 0.19 N m. So PX4's roll saturates at 0.62 of a unit command, pitch at
  0.38, yaw at 0.27.
- motor lag 0.019 s, the mean of the rotor speed's up and down time constants.
- no extra rate-setpoint lag (``rate_lag_s`` 0): with these numbers the env's 100 Hz rate
  loop matches a per-rotor model of PX4's 250 Hz loop on the x500 (63 % of a roll step in
  60 to 65 ms, pitch 56 to 60 ms, yaw about 300 ms). The placeholder with its 100 to 200 ms
  setpoint lag took 190 to 270 ms in roll: the x500 answers 3 to 4 times faster.

Not modelled: the rotors' drag (gz rotorDragCoefficient, about 0.25 N per m/s at hover,
against the wall plugin's own drag force) and the dependence of the torque limits on the
collective thrust.

``gz_x500_sitl``: ``gz_x500`` as the Gazebo benchmark flies it through PX4 v1.17 and the
module (docs/decisions.md 2026-10-23; the per-tick traces of v6 and v8 in Gazebo showed each
piece):

- ``MC_YAW_TQ_CUTOFF`` 2 Hz on the yaw torque and ``IMU_GYRO_CUTOFF`` 40 Hz on the gyro the
  rate loop and the module read (``envs/sims/rate_pid.py``): with them a yaw-rate command
  answers with gain 0.27 and 130 degrees of lag at 1.6 Hz, as in Gazebo, where the env
  without them gave 0.32 and 80 degrees; the policy reads the filtered gyro one tick old,
  which stands in for SITL's actuator latency (``env.py`` ``_observe``);
- ``dt`` 0.012 s: the module runs on gyro callbacks once 10 ms have passed, and the Gazebo
  gyro comes at 250 Hz, so it ticks every 12 ms (83 Hz);
- ``vision_offset_down_m`` 0.24: the bridge sends the pose of the x500 model frame, 0.24 m
  below ``base_link`` (``x500_base/model.sdf``), and EKF2 has no lever-arm correction
  (``EKF2_EV_POS_*`` 0); a fit to the traces put the tracked point 0.18 m below the IMU;
- ``fusion`` on with the bridge's EKF2 settings (``sim2real/estimate_bridge/ekf2_ev.params``):
  the module reads EKF2's estimate, not the fix; with EKF2's resets, so an estimate jump the
  gate rejects reaches the module as a step 1 s later, and every dropout over 0.4 s ends in
  one (``resets``, from 2026-10-23 06:30; earlier runs had them off).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from envs.sims.torch_quad import GRAVITY

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- gz x500 sources
X500_BASE_MASS = 2.0
X500_BASE_INERTIA = (0.02166666666666667, 0.02166666666666667, 0.04000000000000001)
X500_ROTOR_MASS = 0.016076923076923075
X500_ROTOR_INERTIA = (3.8464910483993325e-07, 2.6115851691700804e-05, 2.649858234714004e-05)
X500_MOTOR_CONSTANT = 8.54858e-06  # N / (rad/s)^2
X500_MOMENT_CONSTANT = 0.016  # m: yaw moment per newton of rotor thrust
X500_TAU_UP, X500_TAU_DOWN = 0.0125, 0.025  # s, rotor speed
X500_ESC_RAD_S = (150.0, 1000.0)  # SIM_GZ_EC_MIN, SIM_GZ_EC_MAX
# PX4 motor order, body FRD: (x, y) of the SDF rotor and spin (+1 counter-clockwise seen
# from above, which pushes the body's yaw toward +z FRD).
X500_ROTORS = ((0.174, 0.174, 1.0), (-0.174, -0.174, 1.0), (0.174, -0.174, -1.0),
               (-0.174, 0.174, -1.0))
# 4001_gz_x500: CA_ROTORi_PX, _PY, _KM (CA_ROTORi_CT stays at its default).
X500_CA_ROTORS = ((0.13, 0.22, 0.05), (-0.13, -0.20, 0.05), (0.13, -0.22, -0.05),
                  (-0.13, 0.20, -0.05))
PX4_TUNED_V1_RATE_GAINS = {  # roll, pitch, yaw: MC_*RATE_P / _I / _D
    "p": (0.1466, 0.1102, 0.20),
    "i": (0.1958, 0.1466, 0.10),
    "d": (0.0029, 0.00212, 0.0),
}
MODULE_THRUST_RANGE = (0.08, 0.9)


def px4_allocation(ca_rotors: Any, ct: float = 6.5) -> np.ndarray:
    """PX4's normalised mix for rotors ``(px, py, km)`` (body FRD): (num_rotors, 6) motor
    commands per unit of (roll, pitch, yaw torque, thrust x, y, z), as
    ``ActuatorEffectivenessRotors`` and ``ControlAllocationPseudoInverse`` compute it."""
    axis = np.array([0.0, 0.0, -1.0])
    eff = np.zeros((6, len(ca_rotors)))
    for i, (px, py, km) in enumerate(ca_rotors):
        eff[:3, i] = ct * np.cross(np.array([px, py, 0.0]), axis) - ct * km * axis
        eff[3:, i] = ct * axis
    mix = np.linalg.pinv(eff)
    scale = np.ones(6)

    def roll_pitch(col: np.ndarray) -> float:
        n = int((np.abs(col) > 1e-3).sum())
        return math.sqrt(float((col ** 2).sum()) / (n / 2.0)) if n else 1.0

    scale[0] = scale[1] = max(roll_pitch(mix[:, 0]), roll_pitch(mix[:, 1]))
    scale[2] = float(mix[:, 2].max())
    for k in (5, 4, 3):
        col = np.abs(mix[:, k])
        n = int((col > 1e-7).sum())
        scale[k] = float(col.sum()) / n if n else scale[5]
    return mix / scale


def gz_x500() -> dict[str, Any]:
    """The ``gz_x500`` config mapping, derived from the constants above (module docstring)."""
    r_xy = 0.5 * (X500_ROTOR_INERTIA[0] + X500_ROTOR_INERTIA[1])  # spinning rotor
    arm2 = np.array([[x * x, y * y] for x, y, _ in X500_ROTORS])
    mass = X500_BASE_MASS + 4 * X500_ROTOR_MASS
    inertia = (
        X500_BASE_INERTIA[0] + float(X500_ROTOR_MASS * arm2[:, 1].sum()) + 4 * r_xy,
        X500_BASE_INERTIA[1] + float(X500_ROTOR_MASS * arm2[:, 0].sum()) + 4 * r_xy,
        X500_BASE_INERTIA[2] + float(X500_ROTOR_MASS * arm2.sum()) + 4 * X500_ROTOR_INERTIA[2],
    )
    w_min, w_max = X500_ESC_RAD_S
    esc_min_frac = w_min / w_max

    def rotor_thrust(u: np.ndarray) -> np.ndarray:
        w = w_min + (w_max - w_min) * np.clip(u, 0.0, 1.0)
        return X500_MOTOR_CONSTANT * w * w

    max_thrust = 4 * float(rotor_thrust(np.ones(1))[0])
    u_hover = (math.sqrt(mass * GRAVITY / max_thrust) - esc_min_frac) / (1 - esc_min_frac)
    dthrust_du = 2 * X500_MOTOR_CONSTANT * (w_min + (w_max - w_min) * u_hover) * (w_max - w_min)
    rotors = np.array(X500_ROTORS)

    def torque(t: np.ndarray) -> np.ndarray:  # body torque FRD of per-rotor thrusts
        return np.array([-(rotors[:, 1] * t).sum(), (rotors[:, 0] * t).sum(),
                         (rotors[:, 2] * X500_MOMENT_CONSTANT * t).sum()])

    mix = px4_allocation(X500_CA_ROTORS)
    hover = np.full(4, u_hover)
    per_unit, limit = [], []
    for k in range(3):
        col = mix[:, k]
        per_unit.append(abs(float(torque(dthrust_du * col)[k])))
        room = min((1.0 - u_hover) / c if c > 0 else -u_hover / c for c in col if abs(c) > 1e-9)
        limit.append(abs(float(torque(rotor_thrust(hover + room * col))[k]
                               - torque(rotor_thrust(hover))[k])))
    return {
        "quad": {
            "mass": mass,
            "inertia": inertia,
            "max_thrust": max_thrust,
            "max_torque": tuple(limit),
            "motor_tau": 0.5 * (X500_TAU_UP + X500_TAU_DOWN),
        },
        "rate_pid": dict(PX4_TUNED_V1_RATE_GAINS),
        "actuation": {
            "torque_per_unit": tuple(per_unit),
            "esc_min_frac": esc_min_frac,
            "thrust_cmd_range": MODULE_THRUST_RANGE,
        },
        "rate_lag_s": (0.0, 0.0),
    }


def gz_x500_sitl() -> dict[str, Any]:
    """``gz_x500`` plus PX4's filters, the module's Gazebo tick, the bridge's lever arm and
    EKF2 (module docstring)."""
    cfg = gz_x500()
    cfg["rate_pid"] = {**cfg["rate_pid"], "gyro_cutoff_hz": 40.0, "yaw_torque_cutoff_hz": 2.0}
    cfg["dt"] = 0.012
    cfg["vision_offset_down_m"] = 0.24
    cfg["fusion"] = {"enabled": True, "vision_rate_hz": 50.0, "ev_delay_s": 0.1,
                     "pos_noise_m": 0.1, "vel_noise_mps": 0.1, "gate_sd": 5.0, "resets": True}
    return cfg


AIRFRAMES = {
    "placeholder": lambda: {},
    "gz_x500": gz_x500,
    "gz_x500_sitl": gz_x500_sitl,
}


def _tuples(value: Any) -> Any:
    """YAML lists back to the tuples the config dataclasses hold."""
    if isinstance(value, dict):
        return {k: _tuples(v) for k, v in value.items()}
    if isinstance(value, list):
        return tuple(_tuples(v) for v in value)
    return value


def airframe_config(name: str) -> dict[str, Any]:
    """The config mapping of a named airframe (a fresh copy), or of a YAML file of the same
    shape (a name ending in ``.yaml``, relative to the repo root unless absolute), such as
    the ``airframe.yaml`` that ``eval/fit_airframe.py`` writes for a customer's drone."""
    if name.endswith((".yaml", ".yml")):
        import yaml

        path = Path(name) if Path(name).is_absolute() else REPO_ROOT / name
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected an airframe mapping")
        return _tuples(data)
    if name not in AIRFRAMES:
        raise ValueError(f"unknown airframe {name!r}; known: {sorted(AIRFRAMES)}")
    return AIRFRAMES[name]()
