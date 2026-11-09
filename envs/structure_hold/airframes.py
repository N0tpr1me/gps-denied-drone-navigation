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
  one (``resets``, from 2026-10-23 06:30; earlier runs had them off); height from the
  barometer (``height_source`` baro, decided 2026-11-03), as the benchmark's EKF2 flies it
  since 2026-11-01 (``EKF2_HGT_REF`` 0; ``docs/baseline.md`` "The mock fusion on barometer
  height"), so a vision reset or dropout never moves the estimated height. Every policy up
  to v22 trained and was gated on vision height; the check re-run on the barometer is in
  ``docs/gazebo_check.md``.

``small_300g``: a 300 g class quadrotor (VOXL 2 Mini size), for scoring a policy on a drone
much smaller than the x500 (``docs/airframe-sizes.md``). Its published numbers are those of
ModalAI's Starling 2: 285 g take-off weight, 230 mm motor-to-motor diagonal, 120 mm props,
about 185 g of thrust per motor (thrust/weight 2.6) (Starling 2 datasheet and ModalAI forum
topic 5202, 2026). The rest is scaled from the x500 and stated as an assumption: base
inertia by mass and arm length squared (as ``eval/fit_airframe.py`` guesses an inertia),
rotor inertia and yaw moment constant by prop size, rotor speed linear in the command from
15 % of full speed (as the x500's ESCs), rotor lag 8 ms up and 16 ms down (smaller motors
than the x500's 12.5 / 25 ms), PX4's ``CA_ROTOR*`` at the true rotor positions. The rate
gains are what PX4's autotune would give it: autotune aims at the same closed-loop rise
time (``MC_AT_RISE_TIME``) on any airframe, so each axis keeps the x500's loop gain (rate P
times torque per unit over inertia), with I and D scaled alike. It flies through the same
PX4 pieces as ``gz_x500_sitl`` (filters, 83 Hz tick, EKF2 on the barometer) so that only the
vehicle differs, with the tracking camera 0.02 m below the IMU (on the flight board, not a
0.24 m model offset) and the wind's drag scaled to its frontal area (motor-to-motor
diagonal squared: 0.22 of the x500's). A partner's real drone replaces all of this with
``eval/fit_airframe.py``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from envs.disturbance import NOMINAL_K_QUAD
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


def quad_from_rotors(*, base_mass: float, base_inertia: Any, rotor_mass: float,
                     rotor_inertia: Any, motor_constant: float, moment_constant: float,
                     tau_up: float, tau_down: float, esc_rad_s: Any, rotors: Any,
                     ca_rotors: Any, rate_gains: dict[str, Any]) -> dict[str, Any]:
    """The env mapping of a quadrotor flown through PX4 v1.17, derived from its Gazebo-style
    rotor model and PX4's own allocation geometry (``CA_ROTOR*``) as the module docstring
    explains for the x500: mass and inertia of base plus rotors, max thrust and the thrust
    curve, torque per unit of PX4's normalised torque linearised at hover, the torque limits
    where the first rotor saturates, and the mean rotor lag."""
    r_xy = 0.5 * (rotor_inertia[0] + rotor_inertia[1])  # spinning rotor
    arm2 = np.array([[x * x, y * y] for x, y, _ in rotors])
    mass = base_mass + 4 * rotor_mass
    inertia = (
        base_inertia[0] + float(rotor_mass * arm2[:, 1].sum()) + 4 * r_xy,
        base_inertia[1] + float(rotor_mass * arm2[:, 0].sum()) + 4 * r_xy,
        base_inertia[2] + float(rotor_mass * arm2.sum()) + 4 * rotor_inertia[2],
    )
    w_min, w_max = esc_rad_s
    esc_min_frac = w_min / w_max

    def rotor_thrust(u: np.ndarray) -> np.ndarray:
        w = w_min + (w_max - w_min) * np.clip(u, 0.0, 1.0)
        return motor_constant * w * w

    max_thrust = 4 * float(rotor_thrust(np.ones(1))[0])
    u_hover = (math.sqrt(mass * GRAVITY / max_thrust) - esc_min_frac) / (1 - esc_min_frac)
    dthrust_du = 2 * motor_constant * (w_min + (w_max - w_min) * u_hover) * (w_max - w_min)
    rot = np.array(rotors)

    def torque(t: np.ndarray) -> np.ndarray:  # body torque FRD of per-rotor thrusts
        return np.array([-(rot[:, 1] * t).sum(), (rot[:, 0] * t).sum(),
                         (rot[:, 2] * moment_constant * t).sum()])

    mix = px4_allocation(ca_rotors)
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
            "motor_tau": 0.5 * (tau_up + tau_down),
        },
        "rate_pid": dict(rate_gains),
        "actuation": {
            "torque_per_unit": tuple(per_unit),
            "esc_min_frac": esc_min_frac,
            "thrust_cmd_range": MODULE_THRUST_RANGE,
        },
        "rate_lag_s": (0.0, 0.0),
    }


def gz_x500() -> dict[str, Any]:
    """The ``gz_x500`` config mapping, derived from the constants above (module docstring)."""
    return quad_from_rotors(
        base_mass=X500_BASE_MASS, base_inertia=X500_BASE_INERTIA, rotor_mass=X500_ROTOR_MASS,
        rotor_inertia=X500_ROTOR_INERTIA, motor_constant=X500_MOTOR_CONSTANT,
        moment_constant=X500_MOMENT_CONSTANT, tau_up=X500_TAU_UP, tau_down=X500_TAU_DOWN,
        esc_rad_s=X500_ESC_RAD_S, rotors=X500_ROTORS, ca_rotors=X500_CA_ROTORS,
        rate_gains=PX4_TUNED_V1_RATE_GAINS)


def gz_x500_sitl() -> dict[str, Any]:
    """``gz_x500`` plus PX4's filters, the module's Gazebo tick, the bridge's lever arm and
    EKF2 (module docstring)."""
    cfg = gz_x500()
    cfg["rate_pid"] = {**cfg["rate_pid"], "gyro_cutoff_hz": 40.0, "yaw_torque_cutoff_hz": 2.0}
    cfg["dt"] = 0.012
    cfg["vision_offset_down_m"] = 0.24
    cfg["fusion"] = {"enabled": True, "vision_rate_hz": 50.0, "ev_delay_s": 0.1,
                     "pos_noise_m": 0.1, "vel_noise_mps": 0.1, "gate_sd": 5.0, "resets": True,
                     "height_source": "baro"}
    return cfg


# ---------------------------------------------------------------- 300 g class (small_300g)
SMALL_MASS = 0.285  # kg, Starling 2 take-off weight
SMALL_DIAGONAL = 0.230  # m, motor to motor
SMALL_PROP_RADIUS = 0.060  # m, 120 mm props
SMALL_ROTOR_THRUST = 0.185 * GRAVITY  # N per rotor at full command
SMALL_ROTOR_MASS = 0.003  # kg, motor bell and prop counted with the rotor (assumed)
SMALL_TAU_UP, SMALL_TAU_DOWN = 0.008, 0.016  # s (assumed: faster than the x500's)
SMALL_ESC_RAD_S = (285.0, 1900.0)  # 15 % idle as the x500; the top speed only scales k
SMALL_VISION_OFFSET_DOWN_M = 0.02  # tracking camera on the flight board, near the IMU
X500_DIAGONAL = 2 * math.hypot(0.174, 0.174)  # m, motor to motor


def small_300g() -> dict[str, Any]:
    """The ``small_300g`` config mapping (module docstring): the vehicle, then the PX4
    pieces of ``gz_x500_sitl``."""
    arm = SMALL_DIAGONAL / (2 * math.sqrt(2))  # x and y of each rotor
    base_mass = SMALL_MASS - 4 * SMALL_ROTOR_MASS
    scale = (base_mass / X500_BASE_MASS) * (arm / 0.174) ** 2
    rotor_scale = (SMALL_ROTOR_MASS / X500_ROTOR_MASS) * (SMALL_PROP_RADIUS / 0.14) ** 2
    rotors = tuple((math.copysign(arm, x), math.copysign(arm, y), d) for x, y, d in X500_ROTORS)
    ca_rotors = tuple((math.copysign(arm, x), math.copysign(arm, y), km)
                      for x, y, km in X500_CA_ROTORS)
    common = dict(
        base_mass=base_mass, base_inertia=tuple(i * scale for i in X500_BASE_INERTIA),
        rotor_mass=SMALL_ROTOR_MASS,
        rotor_inertia=tuple(i * rotor_scale for i in X500_ROTOR_INERTIA),
        motor_constant=SMALL_ROTOR_THRUST / SMALL_ESC_RAD_S[1] ** 2,
        moment_constant=X500_MOMENT_CONSTANT * SMALL_PROP_RADIUS / 0.14,
        tau_up=SMALL_TAU_UP, tau_down=SMALL_TAU_DOWN, esc_rad_s=SMALL_ESC_RAD_S,
        rotors=rotors, ca_rotors=ca_rotors)
    cfg = quad_from_rotors(**common, rate_gains=PX4_TUNED_V1_RATE_GAINS)
    # Autotune's equal rise time: the x500's loop gain per axis, every PID term alike.
    x500 = gz_x500()
    authority = lambda c: np.asarray(c["actuation"]["torque_per_unit"]) / np.asarray(  # noqa: E731
        c["quad"]["inertia"])
    ratio = authority(x500) / authority(cfg)
    cfg["rate_pid"] = {k: tuple(float(g * r) for g, r in zip(v, ratio, strict=True))
                       for k, v in PX4_TUNED_V1_RATE_GAINS.items()}
    sitl = gz_x500_sitl()
    cfg["rate_pid"].update({k: sitl["rate_pid"][k]
                            for k in ("gyro_cutoff_hz", "yaw_torque_cutoff_hz")})
    cfg["dt"] = sitl["dt"]
    cfg["fusion"] = dict(sitl["fusion"])
    cfg["vision_offset_down_m"] = SMALL_VISION_OFFSET_DOWN_M
    area = (SMALL_DIAGONAL / X500_DIAGONAL) ** 2
    cfg["disturbance"] = {"k_quad": (0.8 * NOMINAL_K_QUAD * area, 1.2 * NOMINAL_K_QUAD * area)}
    return cfg


AIRFRAMES = {
    "placeholder": lambda: {},
    "gz_x500": gz_x500,
    "gz_x500_sitl": gz_x500_sitl,
    "small_300g": small_300g,
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
