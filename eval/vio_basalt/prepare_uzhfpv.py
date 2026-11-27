"""Lay out one UZH-FPV flight for Basalt (docs/vio-fit.md, "Third fit").

    python -m eval.vio_basalt.prepare_uzhfpv <flight dir> <calib dir> <out dir>

UZH-FPV (University of Zurich, drone racing, CC BY-NC-SA 3.0) ships each flight as a folder
of PNGs with ``left_images.txt``, ``right_images.txt``, ``imu.txt`` and ``groundtruth.txt``
(Leica total station, the IMU's pose), and its Kalibr calibration in a separate zip per
camera mount. Basalt reads EuRoC's ASL layout, so this writes, under ``<out dir>``:

    mav0/cam0/data.csv, mav0/cam0/data/<ns>.png   symlinks to the flight's left PNGs
    mav0/cam1/...                                  right
    mav0/imu0/data.csv                             ns, gyro xyz, accel xyz
    calib.json                                     Basalt calibration (kb4 = Kalibr equidistant)
    truth.txt                                      TUM ``t x y z qx qy qz qw``, seconds

The camera-to-IMU time shift Kalibr found (``timeshift_cam_imu``, t_imu = t_cam + shift) is
added to the image stamps here, so Basalt runs with ``cam_time_offset_ns`` 0. The images are
Snapdragon Flight stereo, 640x480 at 30 Hz, the IMU at 500 Hz. NumPy and PyYAML only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# The IMU noise Basalt ships for both EuRoC and TUM-VI (basalt/data/euroc_ds_calib.json,
# tumvi_512_ds_calib.json; continuous-time densities). UZH-FPV's own Kalibr imu.yaml says
# 0.1 and 0.05, far above any MEMS datasheet; with those Basalt drifted to 0.69 m ATE on
# indoor_forward_3, with these 0.20 m. One noise setting for every dataset keeps the VIO the
# same across them, so the datasets differ only in what they show it.
BASALT_IMU_NOISE = {"accel_noise_std": 0.016, "gyro_noise_std": 0.000282,
                    "accel_bias_std": 0.001, "gyro_bias_std": 0.0001}


def _quat_xyzw(r: np.ndarray) -> list[float]:
    """Rotation matrix to a unit quaternion (x, y, z, w)."""
    m = r
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q = q / np.linalg.norm(q)
    return (q if q[3] >= 0 else -q).tolist()


def kalibr_to_basalt(camchain: dict, imu: dict) -> tuple[dict, float]:
    # Only the IMU rate is taken from Kalibr's imu.yaml (noise: BASALT_IMU_NOISE).
    """Basalt calibration JSON from Kalibr's camchain-imucam and imu YAML, and the time shift."""
    t_imu_cam, intrinsics, resolution, shifts = [], [], [], []
    for name in ("cam0", "cam1"):
        cam = camchain[name]
        if cam["distortion_model"] != "equidistant" or cam["camera_model"] != "pinhole":
            raise ValueError(f"{name}: expected pinhole-equidistant, got "
                             f"{cam['camera_model']}-{cam['distortion_model']}")
        t_ci = np.array(cam["T_cam_imu"], dtype=float)
        t_ic = np.linalg.inv(t_ci)
        qx, qy, qz, qw = _quat_xyzw(t_ic[:3, :3])
        px, py, pz = t_ic[:3, 3].tolist()
        t_imu_cam.append({"px": px, "py": py, "pz": pz, "qx": qx, "qy": qy, "qz": qz, "qw": qw})
        fx, fy, cx, cy = cam["intrinsics"]
        k1, k2, k3, k4 = cam["distortion_coeffs"]
        intrinsics.append({"camera_type": "kb4", "intrinsics": {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy, "k1": k1, "k2": k2, "k3": k3, "k4": k4}})
        resolution.append(list(cam["resolution"]))
        shifts.append(float(cam.get("timeshift_cam_imu", 0.0)))

    calib = {"value0": {
        "T_imu_cam": t_imu_cam,
        "intrinsics": intrinsics,
        "resolution": resolution,
        "calib_accel_bias": [0.0] * 9,
        "calib_gyro_bias": [0.0] * 12,
        "imu_update_rate": float(imu.get("update_rate", 500.0)),
        "accel_noise_std": [BASALT_IMU_NOISE["accel_noise_std"]] * 3,
        "gyro_noise_std": [BASALT_IMU_NOISE["gyro_noise_std"]] * 3,
        "accel_bias_std": [BASALT_IMU_NOISE["accel_bias_std"]] * 3,
        "gyro_bias_std": [BASALT_IMU_NOISE["gyro_bias_std"]] * 3,
        "cam_time_offset_ns": 0,
        "vignette": [],
    }}
    return calib, float(np.mean(shifts))


def _read_images(path: Path) -> list[tuple[float, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        _, stamp, name = line.split()[:3]
        rows.append((float(stamp), name))
    return rows


def _find_one(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"no {pattern} under {folder}")
    return hits[0]


def prepare(flight: Path, calib_dir: Path, out: Path) -> dict:
    camchain = yaml.safe_load(_find_one(calib_dir, "camchain-imucam-*.yaml").read_text())
    imu_yaml = yaml.safe_load(_find_one(calib_dir, "imu.yaml").read_text())
    calib, shift = kalibr_to_basalt(camchain, imu_yaml)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calib.json").write_text(json.dumps(calib, indent=2) + "\n")

    left = _read_images(flight / "left_images.txt")
    right = _read_images(flight / "right_images.txt")
    # Stereo pairs share a stamp; keep only stamps both cameras have.
    right_by_t = {round(t, 6): n for t, n in right}
    pairs = [(t, ln, right_by_t[round(t, 6)]) for t, ln in left if round(t, 6) in right_by_t]
    for cam, idx in (("cam0", 1), ("cam1", 2)):
        data = out / "mav0" / cam / "data"
        data.mkdir(parents=True, exist_ok=True)
        lines = ["#timestamp [ns],filename"]
        for row in pairs:
            ns = int(round((row[0] + shift) * 1e9))
            link = data / f"{ns}.png"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(os.path.relpath(flight / row[idx], data), link)
            lines.append(f"{ns},{ns}.png")
        (out / "mav0" / cam / "data.csv").write_text("\n".join(lines) + "\n")

    imu = np.loadtxt(flight / "imu.txt", comments="#", ndmin=2)  # id t gx gy gz ax ay az
    imu_dir = out / "mav0" / "imu0"
    imu_dir.mkdir(parents=True, exist_ok=True)
    with open(imu_dir / "data.csv", "w") as fh:
        fh.write("#timestamp [ns],w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n")
        for row in imu:
            fh.write(f"{int(round(row[1] * 1e9))}," + ",".join(f"{v:.9f}" for v in row[2:8])
                     + "\n")

    gt = np.loadtxt(flight / "groundtruth.txt", comments="#", ndmin=2)  # t x y z qx qy qz qw
    np.savetxt(out / "truth.txt", gt[:, :8], fmt="%.9f",
               header="timestamp tx ty tz qx qy qz qw")
    return {"frames": len(pairs), "imu": len(imu), "truth": len(gt),
            "truth_s": float(gt[-1, 0] - gt[0, 0]), "time_shift_s": shift}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    info = prepare(Path(args[0]), Path(args[1]), Path(args[2]))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
