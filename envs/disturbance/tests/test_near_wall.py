"""The near-wall flow terms: off by default and exactly as before, and each term's shape."""

import math

import numpy as np
import pytest

from envs.disturbance import (
    DisturbanceConfig,
    air_velocity,
    disturbance_force,
    sample_params,
    shelter_factor,
    wind_velocity,
)

CALM = dict(num_gusts=0, turb_sigma=0.0)
# Wall at x = 0, free air on +x (the model's normal points into the free air).
WALL = dict(wall_point=(0.0, 0.0, 0.0), wall_normal=(1.0, 0.0, 0.0))
FLOW = dict(wall_deflect=(0.2, 0.8), wall_recirc=(0.2, 0.6), wake_ti=(0.1, 0.3),
            shed_amp=(0.1, 0.4))


def at(d: float, n: int) -> np.ndarray:
    return np.tile([d, 0.0, 2.0], (n, 1))


def steady(speed_x: float, n: int = 3, **flow) -> object:
    """A fixed mean wind along x (negative blows into the wall) and nothing else."""
    cfg = DisturbanceConfig(**CALM, mean_speed=abs(speed_x),
                            mean_heading_deg=0.0 if speed_x > 0 else 180.0, **flow)
    p = sample_params(cfg, n, seed=1, **WALL)
    np.testing.assert_allclose(p.mean_wind, np.tile([speed_x, 0.0, 0.0], (n, 1)), atol=1e-12)
    return p


def test_off_by_default_draws_nothing_and_flies_the_same_air():
    cfg = DisturbanceConfig()
    assert not cfg.near_wall_flow and DisturbanceConfig(**FLOW).near_wall_flow
    # No extra draws: the generator ends where it did before the flow existed, wall or not.
    rng_wall, rng_free = np.random.default_rng(3), np.random.default_rng(3)
    p = sample_params(cfg, 64, rng_wall, **WALL)
    sample_params(cfg, 64, rng_free)
    assert rng_wall.bit_generator.state == rng_free.bit_generator.state
    rng = np.random.default_rng(3)
    t, pos = 7.5, rng.uniform(-1.0, 5.0, (64, 3))
    np.testing.assert_array_equal(air_velocity(p, t, pos),
                                  shelter_factor(p, pos)[:, None] * wind_velocity(p, t))


def test_turning_it_on_keeps_every_earlier_draw():
    off = sample_params(DisturbanceConfig(), 16, seed=4, **WALL)
    on = sample_params(DisturbanceConfig(**FLOW), 16, seed=4, **WALL)
    for name in ("mean_wind", "gust_start", "gust_wind", "turb_amp", "turb_phase", "k_quad",
                 "wall_decay", "wall_floor"):
        np.testing.assert_array_equal(getattr(on, name), getattr(off, name), err_msg=name)


def test_free_air_has_no_near_wall_flow():
    p = sample_params(DisturbanceConfig(**FLOW), 8, seed=5)
    rng = np.random.default_rng(5)
    pos = rng.uniform(-3.0, 3.0, (8, 3))
    np.testing.assert_array_equal(air_velocity(p, 4.0, pos), wind_velocity(p, 4.0))


def test_windward_deflection_turns_the_blocked_wind_along_the_wall():
    # 8 m/s into the wall, half the blocked part turned straight up (angle 90).
    p = steady(-8.0, wall_deflect=0.5, wall_deflect_angle_deg=90.0)
    np.testing.assert_allclose(p.wall_tangent, np.tile([0.0, 0.0, 1.0], (3, 1)), atol=1e-12)
    for d in (0.3, 1.0, 2.5):
        g = shelter_factor(p, at(d, 3))
        air = air_velocity(p, 0.0, at(d, 3))
        np.testing.assert_allclose(air[:, 0], -8.0 * g, rtol=1e-12)
        np.testing.assert_allclose(air[:, 2], 0.5 * (1.0 - g) * 8.0, rtol=1e-12)
    # Wind blowing away from the wall is not deflected.
    away = steady(8.0, wall_deflect=0.5, wall_deflect_angle_deg=90.0)
    assert np.allclose(air_velocity(away, 0.0, at(1.0, 3))[:, 2], 0.0)


def test_leeward_recirculation_reverses_flow_and_peaks_at_its_length():
    p = steady(6.0, wall_recirc=0.5, wall_recirc_length=1.5)
    d = np.linspace(0.0, 6.0, 61)
    reverse = np.array([shelter_factor(p, at(di, 3))[0] * 6.0
                        - air_velocity(p, 0.0, at(di, 3))[0, 0] for di in d])
    assert d[np.argmax(reverse)] == pytest.approx(1.5)
    assert reverse.max() == pytest.approx(0.5 * 6.0)
    assert reverse[0] == 0.0  # no flow through the surface itself
    # Wind blowing into the wall has no bubble on this side.
    into = steady(-6.0, wall_recirc=0.5, wall_recirc_length=1.5)
    np.testing.assert_allclose(air_velocity(into, 0.0, at(1.5, 3))[:, 0],
                               -6.0 * shelter_factor(into, at(1.5, 3)), rtol=1e-12)


def test_wake_turbulence_scales_with_the_mean_wind_and_decays_from_the_wall():
    p = steady(10.0, n=200, wake_ti=0.2, wake_length=2.0)
    sigma = np.sqrt((p.wake_amp**2).sum(-1) / 2.0)
    np.testing.assert_allclose(sigma, 0.2 * 10.0, rtol=1e-12)
    # Across the wind (y) only the wake turbulence moves the air at the wall.
    side = np.stack([air_velocity(p, ti, at(0.0, 200))[:, 1] for ti in np.linspace(0, 60, 400)])
    assert side.std() == pytest.approx(2.0, rel=0.1)
    still = np.zeros((200, 3))
    f0 = disturbance_force(p, 3.0, at(0.0, 200), still)
    f_far = disturbance_force(p, 3.0, at(40.0, 200), still)
    assert np.abs(f_far[:, 1]).max() < 1e-6 < np.abs(f0[:, 1]).max()


def test_vortex_shedding_frequency_and_amplitude():
    p = steady(8.0, shed_amp=0.3, shed_width=2.0, shed_strouhal=0.2, wake_length=3.0)
    np.testing.assert_allclose(p.shed_omega, 2.0 * math.pi * 0.2 * 8.0 / 2.0)  # 0.8 Hz
    # Angle 0: horizontal along the wall (the sign of "along" is a convention).
    np.testing.assert_allclose(np.abs(p.shed_dir), np.tile([0.0, 1.0, 0.0], (3, 1)), atol=1e-12)
    t = np.linspace(0.0, 10.0, 2001)
    side = np.array([air_velocity(p, ti, at(1.0, 3))[:, 1] for ti in t])
    np.testing.assert_allclose(np.abs(side).max(0), 0.3 * 8.0 * math.exp(-1.0 / 3.0), rtol=1e-3)
    # In front of the windward face there is no shedding.
    into = steady(-8.0, shed_amp=0.3)
    assert np.all(into.shed_amp == 0.0)


def test_surface_directions_are_in_plane_for_any_surface():
    # A deck underside: the normal points down into the air below it.
    for normal in [(0.0, 0.0, -1.0), (1.0, 0.3, 0.0), (0.2, -1.0, 0.4)]:
        p = sample_params(DisturbanceConfig(**FLOW), 32, seed=6, wall_point=(0, 0, 0),
                          wall_normal=normal)
        for v in (p.wall_tangent, p.shed_dir):
            np.testing.assert_allclose(np.linalg.norm(v, axis=-1), 1.0, rtol=1e-12)
            np.testing.assert_allclose((v * p.wall_normal).sum(-1), 0.0, atol=1e-12)


def test_config_checks_lengths():
    with pytest.raises(ValueError, match="positive"):
        DisturbanceConfig(wake_length=0.0)
