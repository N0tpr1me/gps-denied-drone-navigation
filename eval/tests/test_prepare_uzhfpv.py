"""The UZH-FPV layout for Basalt: Kalibr calibration to Basalt JSON, the EuRoC folders."""

import json

import numpy as np
import yaml

from eval.vio_basalt import prepare_uzhfpv as prep


def _quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _camchain():
    # The UZH-FPV indoor_forward cam0 extrinsic (a camera looking forward, IMU z up).
    t_cam_imu = [[-0.028228787, -0.999601488, 1.2175e-05, 0.021723883],
                 [0.014401252, -0.000418871, -0.999896209, -6.6055e-05],
                 [0.999497744, -0.028225682, 0.014407337, -0.000488176],
                 [0.0, 0.0, 0.0, 1.0]]
    cam = {"T_cam_imu": t_cam_imu, "camera_model": "pinhole",
           "distortion_model": "equidistant", "distortion_coeffs": [-0.01, 0.02, -0.01, 0.002],
           "intrinsics": [278.7, 278.5, 319.8, 242.0], "resolution": [640, 480],
           "timeshift_cam_imu": -0.0167}
    return {"cam0": cam, "cam1": dict(cam, timeshift_cam_imu=-0.0165)}


IMU = {"accelerometer_noise_density": 0.1, "accelerometer_random_walk": 0.002,
       "gyroscope_noise_density": 0.05, "gyroscope_random_walk": 4.0e-05, "update_rate": 500.0}


def test_extrinsic_is_the_inverse_of_kalibrs():
    calib, shift = prep.kalibr_to_basalt(_camchain(), IMU)
    c = calib["value0"]["T_imu_cam"][0]
    r = _quat_to_matrix(c["qx"], c["qy"], c["qz"], c["qw"])
    t_ci = np.array(_camchain()["cam0"]["T_cam_imu"])
    assert np.allclose(r, t_ci[:3, :3].T, atol=1e-6)
    assert np.allclose([c["px"], c["py"], c["pz"]], -t_ci[:3, :3].T @ t_ci[:3, 3], atol=1e-9)
    assert abs(shift + 0.0166) < 1e-9
    assert calib["value0"]["intrinsics"][0]["camera_type"] == "kb4"
    assert calib["value0"]["accel_noise_std"] == [0.016] * 3
    assert calib["value0"]["imu_update_rate"] == 500.0


def test_flight_is_laid_out_as_euroc(tmp_path):
    flight = tmp_path / "flight"
    (flight / "img").mkdir(parents=True)
    for i in range(3):
        for c in (0, 1):
            (flight / "img" / f"image_{c}_{i}.png").write_bytes(b"png")
    (flight / "left_images.txt").write_text(
        "# id timestamp image_name\n" + "".join(
            f"{i} {10 + 0.05 * i:.6f} img/image_0_{i}.png\n" for i in range(3)))
    (flight / "right_images.txt").write_text(
        "# id timestamp image_name\n" + "".join(
            f"{i} {10 + 0.05 * i:.6f} img/image_1_{i}.png\n" for i in range(3)))
    (flight / "imu.txt").write_text("# id t gx gy gz ax ay az\n0 10.0 0 0 0 0 0 9.81\n"
                                    "1 10.002 0 0 0 0 0 9.81\n")
    (flight / "groundtruth.txt").write_text("# t x y z qx qy qz qw\n10.0 1 2 3 0 0 0 1\n"
                                            "10.1 1 2 3 0 0 0 1\n")
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()

    (calib_dir / "camchain-imucam-x.yaml").write_text(yaml.safe_dump(_camchain()))
    (calib_dir / "imu.yaml").write_text(yaml.safe_dump(IMU))

    out = tmp_path / "out"
    info = prep.prepare(flight, calib_dir, out)
    assert info["frames"] == 3 and info["imu"] == 2
    rows = (out / "mav0" / "cam0" / "data.csv").read_text().splitlines()[1:]
    ns = int(rows[0].split(",")[0])
    assert ns == round((10.0 - 0.0166) * 1e9)
    assert (out / "mav0" / "cam1" / "data" / f"{ns}.png").read_bytes() == b"png"
    assert (out / "mav0" / "imu0" / "data.csv").read_text().splitlines()[1].startswith(
        "10000000000,")
    assert np.loadtxt(out / "truth.txt").shape == (2, 8)
    assert json.loads((out / "calib.json").read_text())["value0"]["resolution"][0] == [640, 480]
