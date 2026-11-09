"""A scripted heading error for Gazebo flights: EKF2's heading made wrong by a set angle.

Near steel the compass is off, EKF2 turns its heading with it, and the vision positions stay
in their own (true) frame (``docs/heading-error.md``). The mock models that error directly
(``envs/estimate/heading.py``). In Gazebo it has to reach EKF2, and the compass path cannot
carry it: PX4 v1.17 reloads compass calibration only while disarmed
(``VehicleMagnetometer::ParametersUpdate``), and once EKF2 has aligned its compass in flight
the declination comes from its own earth-field states, which it resets to the World Magnetic
Model whenever it knows where it is, so ``EKF2_MAG_DECL`` moves nothing.

So the heading reaches EKF2 through the vision stream instead: with a ``vision_yaw`` block the
bridge also sends the vehicle's attitude in ODOMETRY (Gazebo's true attitude turned about the
vertical by the scripted error) and ``vision_yaw.params`` turns on EKF2's vision yaw fusion
(``EKF2_EV_CTRL`` 15), which replaces the compass. Positions keep coming in the NED frame they
always did (``MAV_FRAME_LOCAL_NED``), which EKF2 fuses as they are once its yaw is aligned, so
what the controllers get is what the compass case gives them: a heading off by ``error_deg``
over positions in the true frame. EKF2 follows a slow change in the vision yaw; a step larger
than its heading gate (``EKF2_HDG_GATE``) is rejected for a second and then taken as a heading
reset, which the module follows (``docs/decisions.md``, heading resets), so the shipped
profiles use an offset from the start and slow drifts only.

The error is a NED yaw (+ is clockwise seen from above, PX4's sign): ``offset_deg`` from the
bridge's first sample (the compass is already off where the vehicle takes off and engages)
plus a drift of ``drift_rate_deg_s`` from ``drift_at_s`` after the bridge's fault anchor, up
to ``drift_max_deg`` (the vehicle edges toward the steel, or EKF2 gives in to it). The attitude
is the truth at the moment the sample is sent, not delayed with the position: at a hover's
yaw rates the difference is far below the error itself.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from envs.frames import enu_flu_quat_to_ned_frd, matrix_to_quat, quat_to_matrix


@dataclass(frozen=True)
class VisionYaw:
    offset_deg: float = 0.0  # from the first sample, NED yaw (+ clockwise from above)
    drift_at_s: float = 0.0  # after the bridge's fault anchor
    drift_rate_deg_s: float = 0.0  # signed
    drift_max_deg: float = 0.0  # the drift's own size limit

    def __post_init__(self) -> None:
        for name in ("offset_deg", "drift_at_s", "drift_rate_deg_s", "drift_max_deg"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"vision_yaw {name} must be finite")
        if self.drift_at_s < 0 or self.drift_max_deg < 0:
            raise ValueError("vision_yaw drift_at_s and drift_max_deg must be >= 0")
        if abs(self.offset_deg) + self.drift_max_deg > 180:
            raise ValueError("vision_yaw offset plus drift must stay within 180 deg")

    def error_deg(self, t_rel: float | None) -> float:
        """The heading error at ``t_rel`` seconds after the anchor (None: before it)."""
        drift = 0.0
        if t_rel is not None and t_rel > self.drift_at_s:
            drift = self.drift_rate_deg_s * (t_rel - self.drift_at_s)
            drift = max(-self.drift_max_deg, min(self.drift_max_deg, drift))
        return self.offset_deg + drift

    def attitude_ned(self, quat_enu_flu: Any, t_rel: float | None) -> np.ndarray:
        """The attitude to send, ``(w, x, y, z)`` FRD body in NED: the true one (Gazebo's
        ENU/FLU ``quat_enu_flu``) turned about the vertical by ``error_deg(t_rel)``."""
        r = quat_to_matrix(enu_flu_quat_to_ned_frd(np.asarray(quat_enu_flu, dtype=np.float64)))
        e = math.radians(self.error_deg(t_rel))
        c, s = math.cos(e), math.sin(e)
        turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # about NED down
        return matrix_to_quat(turn @ r)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VisionYaw:
        unknown = set(d) - {"offset_deg", "drift_at_s", "drift_rate_deg_s", "drift_max_deg"}
        if unknown:
            raise ValueError(f"unknown vision_yaw keys: {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in d.items()})

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def ned_yaw_deg(q_ned_frd: Any) -> float:
    """The NED yaw of an FRD-in-NED attitude, degrees."""
    r = quat_to_matrix(np.asarray(q_ned_frd, dtype=np.float64))
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))
