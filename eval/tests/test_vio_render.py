"""The rendered-VIO pipeline (eval/vio_render): the flight and its IMU agree, the camera sees
what it should where it should, the conditions do what they say, the dataset files are
what Basalt and eval/vio_fit read, and the three configs fly the same thing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from eval import vio_fit
from eval.vio_render import degrade, euroc
from eval.vio_render.camera import RigConfig, world_from_cam
from eval.vio_render.isaac_backend import light_matrix, usd_camera_matrix
from eval.vio_render.raster import RasterRenderer
from eval.vio_render.scene import build_scene, concrete_texture, wall_scene
from eval.vio_render.trajectory import (
    FlightConfig,
    G,
    attitude,
    log_so3,
    quat_from_rot,
    rot_from_quat,
    simulate,
)

REPO = Path(__file__).resolve().parents[2]
QUIET = dict(accel_noise_density=0.0, gyro_noise_density=0.0, accel_bias_walk=0.0,
             gyro_bias_walk=0.0, accel_bias_std=0.0, gyro_bias_std=0.0, vib_accel_rms=0.0,
             vib_gyro_rms=0.0)


@pytest.fixture(scope="module")
def flight():
    return simulate(FlightConfig(hold_s=12.0, **QUIET), seed=3)


@pytest.fixture(scope="module")
def small_scene():
    return wall_scene(width_m=6.0, height_m=4.0, texels_per_m=64.0, seed=2)


def test_quaternion_round_trip():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((50, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= np.where(q[:, :1] < 0, -1, 1)
    np.testing.assert_allclose(quat_from_rot(rot_from_quat(q)), q, atol=1e-9)


def test_attitude_points_thrust_along_body_z():
    acc = np.array([[0.5, -0.3, 0.2], [0.0, 0.0, 0.0]])
    r = attitude(acc, np.array([0.3, -1.0]))
    for k in range(2):
        np.testing.assert_allclose(r[k].T @ r[k], np.eye(3), atol=1e-12)
        assert np.linalg.det(r[k]) == pytest.approx(1.0)
        f = acc[k] + [0, 0, G]
        np.testing.assert_allclose(r[k][:, 2], f / np.linalg.norm(f), atol=1e-12)


def test_flight_sits_still_then_holds(flight):
    still = flight.t < 3.0
    np.testing.assert_allclose(flight.pos[still], np.tile(flight.pos[0], (still.sum(), 1)),
                               atol=1e-12)
    np.testing.assert_allclose(flight.accel[still], np.tile([0, 0, G], (still.sum(), 1)),
                               atol=1e-9)
    hold = flight.t > flight.hold_start_s + 2.0
    err = np.linalg.norm(flight.pos[hold] - flight.hold_point, axis=1)
    assert np.percentile(err, 95) < 0.4  # holding in gusts, not wandering off
    assert np.abs(flight.gyro_true).max() < 1.5  # a hold, not aerobatics


def test_imu_integrates_back_to_the_truth(flight):
    # Dead-reckon the noise-free IMU from the first sample: it must land on the truth
    # (the IMU samples are period means, so nothing aliases).
    dt = float(flight.t[1] - flight.t[0])
    r = rot_from_quat(flight.quat[:1])[0]
    p, v = flight.pos[0].copy(), flight.vel[0].copy()
    gvec = np.array([0.0, 0.0, G])
    for i in range(flight.t.size - 1):
        w = flight.gyro[i] * dt
        th = np.linalg.norm(w)
        rn = r
        if th > 0:
            k = w / th
            kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            rn = r @ (np.eye(3) + math.sin(th) * kx + (1 - math.cos(th)) * kx @ kx)
        a = 0.5 * (r + rn) @ flight.accel[i] - gvec
        p, v, r = p + v * dt + 0.5 * a * dt * dt, v + a * dt, rn
    assert np.linalg.norm(p - flight.pos[-1]) < 0.05
    rt = rot_from_quat(flight.quat[-1:])[0]
    assert np.degrees(np.linalg.norm(log_so3((r.T @ rt)[None])[0])) < 0.2


def test_rig_looks_at_the_wall():
    rig = RigConfig(pitch_deg=0.0)
    r, p = rig.body_from_cam(0)
    np.testing.assert_allclose(r[:, 2], [1, 0, 0], atol=1e-12)  # optical axis = body x
    np.testing.assert_allclose(r[:, 0], [0, -1, 0], atol=1e-12)  # image right = body right
    assert p[1] > 0 > rig.body_from_cam(1)[1][1]  # cam0 is the left camera
    # A point straight ahead projects to the principal point.
    uv, z = rig.project(np.array([[0.0, 0.0, 2.0]]))
    np.testing.assert_allclose(uv[0], [rig.cx, rig.cy])


def test_render_range_matches_the_geometry(small_scene):
    rig = RigConfig(width=160, height=120, pitch_deg=0.0)
    ren = RasterRenderer(small_scene, rig)
    rot_wc, centre = world_from_cam(rig, 0, np.array([-1.5, 0.0, 0.0]), np.eye(3))
    rad, rng = ren.render(rot_wc, centre)
    assert rad.shape == rng.shape == (120, 160)
    # The centre pixel looks straight at the wall 1.5 m - lever away.
    assert rng[60, 80] == pytest.approx(1.5 - rig.lever_m, abs=0.01)
    # Off-axis range grows as 1/cos.
    d = np.array([(0 - rig.cx) / rig.fx, 0.0, 1.0])
    assert rng[int(round(rig.cy)), 0] == pytest.approx((1.5 - rig.lever_m) * np.linalg.norm(d),
                                                       rel=0.01)
    assert 0.1 < float(rad.mean()) < 1.0


def test_texture_is_trackable():
    tex = concrete_texture((256, 256), seed=1)
    assert 0.45 < float(tex.mean()) < 0.55
    # Plenty of fine detail: neighbouring texels differ, not only broad stains.
    assert float(np.abs(np.diff(tex, axis=1)).mean()) > 0.02


def test_fog_keeps_near_contrast_and_greys_the_far():
    rad = np.array([[0.2, 0.8], [0.2, 0.8]])
    rng = np.array([[1.0, 1.0], [30.0, 30.0]])
    out = degrade.fog(rad, rng, visibility_m=8.0, airlight=0.7)
    near, far = out[0, 1] - out[0, 0], out[1, 1] - out[1, 0]
    assert near == pytest.approx(0.6 * math.exp(-3.912 / 8.0), rel=1e-6)
    assert far < 0.01 and out[1, 0] == pytest.approx(0.7, abs=0.01)


def test_dust_is_the_same_grains_for_both_cameras(small_scene):
    rig = RigConfig(width=160, height=120)
    cond = degrade.Condition("dust", dust_per_m3=500.0)
    cloud = degrade.DustCloud(cond, (4.0, 0.0), seed=1)
    ren = RasterRenderer(small_scene, rig)
    pos, rwb = np.array([-1.5, 0.0, 0.0]), np.eye(3)
    outs = []
    for cam in (0, 1):
        rot_wc, centre = world_from_cam(rig, cam, pos, rwb)
        rad, rng = ren.render(rot_wc, centre)
        outs.append(cloud.draw(rad, rng, rig, rot_wc, centre, 1.0, 1.0) - rad)
    # Both views get grains, and the same cloud: comparable amounts of added light.
    a, b = (float(np.abs(o).sum()) for o in outs)
    assert a > 0 and b > 0 and 0.5 < a / b < 2.0
    # Grains drift with the wind.
    p0 = cloud.positions(0.0, pos)
    p1 = cloud.positions(0.1, pos)
    moved = np.linalg.norm(p1 - p0, axis=1)
    assert np.median(moved) == pytest.approx(0.4, rel=0.25)


def test_dust_cloud_arrives_during_the_flight(small_scene):
    cond = degrade.Condition("cloud", dust_per_m3=500.0, haze_visibility_m=10.0,
                             dust_start_s=10.0, dust_ramp_s=4.0)
    assert cond.dust_share(9.9) == 0.0 and cond.dust_share(12.0) == 0.5
    assert cond.dust_share(20.0) == 1.0
    rig = RigConfig(width=160, height=120)
    ren = RasterRenderer(small_scene, rig)
    cloud = degrade.DustCloud(cond, (0.0, 0.0), seed=1)
    rot_wc, centre = world_from_cam(rig, 0, np.array([-1.5, 0.0, 0.0]), np.eye(3))
    rad, rng = ren.render(rot_wc, centre)

    def changed(t: float) -> float:
        out = degrade.apply(cond, rad, rng, dust=cloud, rig=rig, rot_wc=rot_wc,
                            centre=centre, t=t)
        return float(np.abs(out - rad).sum())

    assert changed(5.0) == 0.0  # clear air before it arrives
    assert 0.0 < changed(11.0) < changed(15.0)


def test_camera_exposes_for_the_light():
    cam = degrade.CameraModel()
    exp_bright, gain_bright = cam.exposure(0.5)
    exp_dark, gain_dark = cam.exposure(0.005)
    assert gain_bright == 1.0 and exp_bright < 5.0
    assert exp_dark == cam.exposure_ms[1] and gain_dark > 1.0
    rng = np.random.default_rng(0)
    flat = np.full((64, 64), 0.5)
    bright = cam.capture(flat, exp_bright, gain_bright, rng)
    dark = cam.capture(flat * 0.01, exp_dark, gain_dark, rng)
    assert abs(float(bright.mean()) - 0.4 * 255) < 3
    assert float(dark.std()) > 2 * float(bright.std())  # noisier in the dark


def test_motion_blur_smears_along_the_motion():
    img = np.zeros((20, 40))
    img[:, 20] = 1.0
    out = degrade.motion_blur(img, 6.0, 0.0)
    assert out[:, 20].mean() < 0.5 and out[:, 23].mean() > 0
    np.testing.assert_allclose(degrade.motion_blur(img, 0.2, 0.0), img)


def test_png_round_trip():
    img = np.random.default_rng(1).integers(0, 256, (7, 11), dtype=np.uint8)
    np.testing.assert_array_equal(euroc.read_png_gray(euroc.png_gray(img)), img)


def test_dataset_files_read_back(tmp_path, flight):
    rig = RigConfig(width=64, height=48)
    stamps = [euroc.stamp_ns(t) for t in flight.t[flight.cam_index]]
    euroc.finish_dataset(tmp_path, stamps, flight, rig, FlightConfig(),
                         euroc.CalibError(), seed=1)
    truth = vio_fit.load_euroc_csv(tmp_path / "mav0/state_groundtruth_estimate0/data.csv")
    np.testing.assert_allclose(truth.pos, flight.pos, atol=1e-6)
    np.testing.assert_allclose(truth.t, flight.t + 1.0, atol=1e-9)
    imu = np.loadtxt(tmp_path / "mav0/imu0/data.csv", delimiter=",", comments="#")
    np.testing.assert_allclose(imu[:, 4:7], flight.accel, atol=1e-6)
    cam = (tmp_path / "mav0/cam1/data.csv").read_text().splitlines()
    assert len(cam) == len(stamps) + 1 and cam[1] == f"{stamps[0]},{stamps[0]}.png"
    calib = json.loads((tmp_path / "basalt_calib.json").read_text())["value0"]
    assert calib["intrinsics"][0]["camera_type"] == "pinhole"
    # The calibration Basalt is told is near the truth but not equal to it.
    fx = calib["intrinsics"][0]["intrinsics"]["fx"]
    assert fx != rig.fx and abs(fx / rig.fx - 1) == pytest.approx(0.003, rel=1e-6)
    ext = calib["T_imu_cam"][0]
    assert abs(ext["px"] - rig.lever_m) <= 0.0011
    cfg = json.loads((tmp_path / "basalt_config.json").read_text())["value0"]
    assert cfg["config.vio_enforce_realtime"] is False


def test_usd_camera_matrix_flips_to_usd_axes():
    rig = RigConfig(pitch_deg=0.0)
    rot_wc, centre = world_from_cam(rig, 0, np.array([-1.5, 0.0, 0.0]), np.eye(3))
    m = usd_camera_matrix(rot_wc, centre)
    # USD camera looks along its -z (the third row, negated) = world +x, up = world +z.
    np.testing.assert_allclose(-m[2, :3], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(m[1, :3], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(m[3, :3], centre)
    lm = light_matrix(np.array([0.3, 0.2, -0.93]))
    d = -lm[2, :3]
    np.testing.assert_allclose(d, np.array([0.3, 0.2, -0.93]) / np.linalg.norm([0.3, 0.2, -0.93]))


def _params(name: str) -> dict:
    return yaml.safe_load((REPO / "eval/configs" / name).read_text())["params"]


def test_configs_fly_the_same_thing():
    raster = _params("vio_render_wall.yaml")
    for name in ("vio_render_wall_isaac.yaml", "vio_render_pool.yaml"):
        other = _params(name)
        for key in ("scene", "rig", "flight", "camera", "conditions"):
            assert other.get(key) == raster.get(key), f"{name}: {key} differs"
    names = [c["name"] for c in raster["conditions"]]
    assert names[0] == "clear" and len(set(names)) == len(names)
    for c in raster["conditions"]:
        degrade.Condition.from_dict(c)
    build_scene(raster["scene"])
    assert not any(9000 <= s <= 9999 for s in raster["seeds"])


def test_queue_waits_for_isaac_parity():
    from training.queue import load_queue

    q = load_queue(REPO / "training/queues/vio_render.yaml")
    jobs = {j.id: j for j in q.jobs}
    for jid in ("isaac_render_s1", "isaac_render_s2"):
        assert jobs[jid].after == ["isaac_ok"]
    assert "isaac_parity" in jobs["isaac_ok"].cmd
    # Never more than two GPU jobs at once: the third waits for the first.
    assert jobs["isaac_render_s3"].after == ["isaac_render_s1"]
