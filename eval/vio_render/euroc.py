"""Write a rendered flight in the EuRoC ASL layout, plus Basalt's calibration and config.

The layout is the one Basalt's ``--dataset-type euroc`` reads and ``eval/vio_fit.py``
already scores (docs/vio-fit.md, the second fit):

    <root>/mav0/cam0/data.csv, cam0/data/<ns>.png     left camera, 8-bit grey
    <root>/mav0/cam1/data.csv, cam1/data/<ns>.png     right camera, same stamps
    <root>/mav0/imu0/data.csv                         gyro (rad/s), accel (m/s^2), body
    <root>/mav0/state_groundtruth_estimate0/data.csv  truth: p, q (w x y z), v, biases
    <root>/basalt_calib.json, <root>/basalt_config.json

PNGs are written with zlib only (no imaging library), so this stays NumPy + stdlib.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.vio_render.camera import RigConfig
from eval.vio_render.trajectory import Flight, FlightConfig, quat_from_rot

T0_NS = 1_000_000_000  # dataset clock at the flight's t = 0 (EuRoC stamps are never 0)

# Basalt's own EuRoC config at the pinned commit (eval/vio_basalt/Dockerfile), unchanged
# except that it never drops frames to keep up (offline, every frame is processed).
BASALT_CONFIG = {
    "config.optical_flow_type": "frame_to_frame",
    "config.optical_flow_detection_grid_size": 50,
    "config.optical_flow_max_recovered_dist2": 0.04,
    "config.optical_flow_pattern": 51,
    "config.optical_flow_max_iterations": 5,
    "config.optical_flow_epipolar_error": 0.005,
    "config.optical_flow_levels": 3,
    "config.optical_flow_skip_frames": 1,
    "config.vio_linearization_type": "ABS_QR",
    "config.vio_sqrt_marg": True,
    "config.vio_max_states": 3,
    "config.vio_max_kfs": 7,
    "config.vio_min_frames_after_kf": 5,
    "config.vio_new_kf_keypoints_thresh": 0.7,
    "config.vio_debug": False,
    "config.vio_extended_logging": False,
    "config.vio_obs_std_dev": 0.5,
    "config.vio_obs_huber_thresh": 1.0,
    "config.vio_min_triangulation_dist": 0.05,
    "config.vio_outlier_threshold": 3.0,
    "config.vio_filter_iteration": 4,
    "config.vio_max_iterations": 7,
    "config.vio_enforce_realtime": False,
    "config.vio_use_lm": True,
    "config.vio_lm_lambda_initial": 1e-4,
    "config.vio_lm_lambda_min": 1e-6,
    "config.vio_lm_lambda_max": 1e2,
    "config.vio_lm_landmark_damping_variant": 1,
    "config.vio_lm_pose_damping_variant": 1,
    "config.vio_scale_jacobian": False,
    "config.vio_init_pose_weight": 1e8,
    "config.vio_init_ba_weight": 1e1,
    "config.vio_init_bg_weight": 1e2,
    "config.vio_marg_lost_landmarks": True,
    "config.vio_kf_marg_feature_ratio": 0.1,
    "config.mapper_obs_std_dev": 0.25,
    "config.mapper_obs_huber_thresh": 1.5,
    "config.mapper_detection_num_points": 800,
    "config.mapper_num_frames_to_match": 30,
    "config.mapper_frames_to_match_threshold": 0.04,
    "config.mapper_min_matches": 20,
    "config.mapper_ransac_threshold": 5e-5,
    "config.mapper_min_track_length": 5,
    "config.mapper_max_hamming_distance": 70,
    "config.mapper_second_best_test_ratio": 1.2,
    "config.mapper_bow_num_bits": 16,
    "config.mapper_min_triangulation_dist": 0.07,
    "config.mapper_no_factor_weights": False,
    "config.mapper_use_factors": True,
    "config.mapper_use_lm": True,
    "config.mapper_lm_lambda_min": 1e-32,
    "config.mapper_lm_lambda_max": 1e3,
}


def png_gray(img: np.ndarray) -> bytes:
    """An 8-bit greyscale PNG of ``img`` (H, W) uint8."""
    if img.dtype != np.uint8 or img.ndim != 2:
        raise ValueError("png_gray wants a 2-D uint8 image")
    h, w = img.shape
    raw = np.empty((h, w + 1), dtype=np.uint8)
    raw[:, 0] = 0  # filter: none
    raw[:, 1:] = img

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 1)) + chunk(b"IEND", b""))


def read_png_gray(data: bytes) -> np.ndarray:
    """Inverse of ``png_gray`` (filter type 0 only), for tests."""
    pos = 8
    idat = b""
    w = h = 0
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + n]
        if kind == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, w + 1)
    return raw[:, 1:].copy()


def stamp_ns(t_s: float) -> int:
    return T0_NS + int(round(t_s * 1e9))


@dataclass(frozen=True)
class CalibError:
    """What the VIO is told differs from the truth, as on a real drone after a factory
    calibration: focal length off by ``focal_frac``, each camera's mounting rotation off by
    ``rot_deg`` and position by ``pos_m`` (random directions, fixed per seed), and the
    camera stamps late by ``time_offset_ms`` against the IMU (an unknown fixed offset)."""

    focal_frac: float = 0.003
    rot_deg: float = 0.1
    pos_m: float = 0.001
    time_offset_ms: float = 2.0

    @classmethod
    def from_dict(cls, data: dict) -> CalibError:
        return cls(**data)


def finish_dataset(root: Path, stamps: list[int], flight: Flight, rig: RigConfig,
                   fcfg: FlightConfig, err: CalibError | None = None, seed: int = 0) -> None:
    """Everything but the images (written frame by frame by the renderer): the camera
    lists, IMU, truth and Basalt's two files."""
    mav = Path(root) / "mav0"
    for cam in ("cam0", "cam1"):
        lines = ["#timestamp [ns],filename"] + [f"{ns},{ns}.png" for ns in stamps]
        (mav / cam).mkdir(parents=True, exist_ok=True)
        (mav / cam / "data.csv").write_text("\n".join(lines) + "\n")
    write_imu(mav / "imu0" / "data.csv", flight)
    write_truth(mav / "state_groundtruth_estimate0" / "data.csv", flight)
    write_basalt_files(Path(root), rig, fcfg, err, seed)


def write_imu(path: Path, flight: Flight) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
            "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"]
    for t, g, a in zip(flight.t, flight.gyro, flight.accel, strict=True):
        rows.append(f"{stamp_ns(t)},{g[0]:.9f},{g[1]:.9f},{g[2]:.9f},"
                    f"{a[0]:.9f},{a[1]:.9f},{a[2]:.9f}")
    path.write_text("\n".join(rows) + "\n")


def write_truth(path: Path, flight: Flight) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], "
            "q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], "
            "b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], "
            "b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]"]
    for t, p, q, v in zip(flight.t, flight.pos, flight.quat, flight.vel, strict=True):
        rows.append(f"{stamp_ns(t)},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},{q[0]:.8f},{q[1]:.8f},"
                    f"{q[2]:.8f},{q[3]:.8f},{v[0]:.6f},{v[1]:.6f},{v[2]:.6f},0,0,0,0,0,0")
    path.write_text("\n".join(rows) + "\n")


def _small_rot(axis_angle: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(axis_angle))
    if th < 1e-12:
        return np.eye(3)
    k = axis_angle / th
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * kx @ kx


def basalt_calib(rig: RigConfig, fcfg: FlightConfig, err: CalibError | None = None,
                 seed: int = 0) -> dict:
    """Basalt's calibration for the rig: pinhole cameras, T_imu_cam, IMU noise.

    The IMU noise Basalt is told is the simulated density times two, the usual margin for a
    VIO's noise settings (EuRoC's Basalt calibration inflates its datasheet values too)."""
    err = err or CalibError(0.0, 0.0, 0.0, 0.0)
    rng = np.random.default_rng([seed, 0xCA])
    cams = []
    for i in (0, 1):
        r, p = rig.body_from_cam(i)
        d = rng.standard_normal(3)
        r = _small_rot(np.radians(err.rot_deg) * d / np.linalg.norm(d)) @ r
        d = rng.standard_normal(3)
        p = p + err.pos_m * d / np.linalg.norm(d)
        q = quat_from_rot(r[None])[0]
        cams.append({"px": float(p[0]), "py": float(p[1]), "pz": float(p[2]),
                     "qx": float(q[1]), "qy": float(q[2]), "qz": float(q[3]),
                     "qw": float(q[0])})
    f = rig.fx * (1.0 + err.focal_frac * (1 if rng.random() < 0.5 else -1))
    intr = {"camera_type": "pinhole",
            "intrinsics": {"fx": f, "fy": f, "cx": rig.cx, "cy": rig.cy}}
    vignette = {"value0": 0, "value1": 50000000000, "value2": [[1.0]] * 15}
    return {"value0": {
        "T_imu_cam": cams,
        "intrinsics": [intr, intr],
        "resolution": [[rig.width, rig.height], [rig.width, rig.height]],
        "vignette": [vignette, vignette],
        "calib_accel_bias": [0.0] * 9,
        "calib_gyro_bias": [0.0] * 12,
        "imu_update_rate": fcfg.imu_hz,
        "accel_noise_std": [2 * fcfg.accel_noise_density] * 3,
        "gyro_noise_std": [2 * fcfg.gyro_noise_density] * 3,
        "accel_bias_std": [2 * fcfg.accel_bias_walk] * 3,
        "gyro_bias_std": [2 * fcfg.gyro_bias_walk] * 3,
        "T_mocap_world": {"px": 0.0, "py": 0.0, "pz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
                          "qw": 1.0},
        "T_imu_marker": {"px": 0.0, "py": 0.0, "pz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
                         "qw": 1.0},
        "mocap_time_offset_ns": 0,
        "mocap_to_imu_offset_ns": 0,
        "cam_time_offset_ns": 0,
    }}


def write_basalt_files(root: Path, rig: RigConfig, fcfg: FlightConfig,
                       err: CalibError | None = None, seed: int = 0) -> None:
    calib = basalt_calib(rig, fcfg, err, seed)
    (root / "basalt_calib.json").write_text(json.dumps(calib, indent=2))
    (root / "basalt_config.json").write_text(
        json.dumps({"value0": BASALT_CONFIG}, indent=2))
