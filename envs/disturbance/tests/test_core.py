import math

import numpy as np
import pytest

from envs.disturbance import (
    NOMINAL_K_QUAD,
    DisturbanceConfig,
    disturbance_force,
    drag_force,
    sample_params,
    shelter_factor,
    wall_distance,
    wind_velocity,
)

CALM = dict(mean_speed=0.0, num_gusts=0, turb_sigma=0.0)


def zeros(n: int) -> np.ndarray:
    return np.zeros((n, 3))


def steady(speed: float, n: int = 4, **overrides) -> np.ndarray:
    """Params with a fixed mean wind blowing along +x."""
    cfg = DisturbanceConfig(**{**CALM, "mean_speed": speed, **overrides})
    p = sample_params(cfg, n, seed=0)
    p.mean_wind[:] = (speed, 0.0, 0.0)
    return p


def test_zero_wind_gives_zero_force():
    p = sample_params(DisturbanceConfig(**CALM), 8, seed=1)
    for t in (0.0, 3.7, 60.0):
        assert np.array_equal(wind_velocity(p, t), zeros(8))
        assert np.array_equal(disturbance_force(p, t, zeros(8), zeros(8)), zeros(8))


def test_drag_points_downwind_and_scales_with_speed_squared():
    f1 = disturbance_force(steady(4.0), 0.0, zeros(4), zeros(4))
    f2 = disturbance_force(steady(8.0), 0.0, zeros(4), zeros(4))
    assert np.all(f1[:, 0] > 0) and np.allclose(f1[:, 1:], 0.0)
    np.testing.assert_allclose(f2, 4.0 * f1, rtol=1e-12)


def test_motion_in_still_air_is_opposed():
    p = steady(0.0)
    vel = np.tile([0.0, 2.0, -1.0], (4, 1))
    f = disturbance_force(p, 0.0, zeros(4), vel)
    speed = np.linalg.norm(vel[0])
    np.testing.assert_allclose(f, -(p.k_quad * speed)[:, None] * vel, rtol=1e-12)


def test_linear_drag_term_scales_linearly():
    v = np.tile([3.0, -4.0, 0.0], (2, 1))
    f = drag_force(v, k_lin=np.full(2, 0.5), k_quad=np.zeros(2))
    np.testing.assert_allclose(f, 0.5 * v)
    np.testing.assert_allclose(drag_force(2 * v, np.full(2, 0.5), np.zeros(2)), 2 * f)


def test_gust_step_fires_at_scheduled_time():
    cfg = DisturbanceConfig(
        **{**CALM, "num_gusts": 1},
        gust_start=5.0,
        gust_duration=1.0,
        gust_speed=3.0,
    )
    p = sample_params(cfg, 3, seed=2)
    times = np.array([4.999, 5.0, 5.999, 6.0, 9.0])
    speeds = [np.linalg.norm(wind_velocity(p, t), axis=-1) for t in times]
    np.testing.assert_allclose(speeds, [[0] * 3, [3] * 3, [3] * 3, [0] * 3, [0] * 3], atol=1e-12)
    # Per-env time: each env at its own clock.
    per_env = np.linalg.norm(wind_velocity(p, np.array([4.0, 5.5, 7.0])), axis=-1)
    np.testing.assert_allclose(per_env, [0.0, 3.0, 0.0], atol=1e-12)


def test_scalar_and_per_env_time_agree():
    p = sample_params(DisturbanceConfig(), 5, seed=3)
    np.testing.assert_array_equal(wind_velocity(p, 12.5), wind_velocity(p, np.full(5, 12.5)))


def test_same_seed_same_wind_different_seed_different_wind():
    cfg = DisturbanceConfig()
    t = np.linspace(0.0, 30.0, 7)
    a = [wind_velocity(sample_params(cfg, 16, seed=42), ti) for ti in t]
    b = [wind_velocity(sample_params(cfg, 16, seed=42), ti) for ti in t]
    c = [wind_velocity(sample_params(cfg, 16, seed=43), ti) for ti in t]
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_turbulence_has_configured_spread_and_correlation_time():
    cfg = DisturbanceConfig(**{**CALM, "turb_sigma": 1.0}, turb_vertical_ratio=0.4, turb_tau=1.5)
    p = sample_params(cfg, 10_000, seed=4)
    w = wind_velocity(p, 17.3)
    np.testing.assert_allclose(w.mean(0), 0.0, atol=0.05)
    np.testing.assert_allclose(w.std(0), [1.0, 1.0, 0.4], rtol=0.05)
    # Gauss-Markov target: correlation exp(-lag / tau) at lag = tau.
    w_lag = wind_velocity(p, 17.3 + 1.5)
    corr = (w[:, 0] * w_lag[:, 0]).mean() / (w[:, 0] * w[:, 0]).mean()
    assert abs(corr - math.exp(-1.0)) < 0.08


def test_turbulence_variance_is_exact_per_env():
    p = sample_params(DisturbanceConfig(turb_sigma=(0.2, 1.4)), 50, seed=5)
    sigma = np.sqrt((p.turb_amp**2).sum(-1) / 2.0)
    np.testing.assert_allclose(sigma[:, 0], sigma[:, 1])
    assert np.all((sigma[:, 0] >= 0.2) & (sigma[:, 0] <= 1.4))


@pytest.mark.parametrize("d, expected", [(5.0, 2.78), (0.5, 0.40)])
def test_nominal_wall_term_reproduces_published_case(d, expected):
    # 10 m/s headwind blowing towards a wall at x = 0 whose free side is +x.
    p = sample_params(
        DisturbanceConfig.nominal(), 2, seed=6, wall_point=(0, 0, 0), wall_normal=(1, 0, 0)
    )
    p.mean_wind[:] = (-10.0, 0.0, 0.0)
    pos = np.array([[d, 0.0, 1.5], [d, -3.0, 4.0]])
    f = disturbance_force(p, 0.0, pos, zeros(2))
    np.testing.assert_allclose(f[:, 0], -expected, rtol=1e-9)
    np.testing.assert_allclose(f[:, 1:], 0.0, atol=1e-12)
    assert np.all(p.k_quad == NOMINAL_K_QUAD)


def test_shelter_factor_limits():
    cfg = DisturbanceConfig(**CALM, wall_floor=0.2)
    walled = sample_params(cfg, 3, seed=7, wall_point=(1, 2, 0), wall_normal=(0, -2, 0))
    pos = np.array([[0.0, 2.0, 1.0], [0.0, 5.0, 1.0], [0.0, -40.0, 1.0]])
    np.testing.assert_allclose(wall_distance(walled, pos), [0.0, -3.0, 42.0])
    # At the surface and behind it: the floor. Far away: free stream.
    np.testing.assert_allclose(shelter_factor(walled, pos), [0.2, 0.2, 1.0], atol=1e-12)
    free = sample_params(cfg, 3, seed=7)
    np.testing.assert_array_equal(shelter_factor(free, pos), np.ones(3))


def test_shelter_falls_monotonically_towards_the_wall():
    p = sample_params(DisturbanceConfig(), 64, seed=8, wall_point=(0, 0, 0), wall_normal=(1, 0, 0))
    d = np.linspace(0.0, 6.0, 25)
    g = np.stack([shelter_factor(p, np.tile([di, 0.0, 1.0], (64, 1))) for di in d])
    assert np.all(np.diff(g, axis=0) >= 0.0)
    assert np.all((g >= 0.0) & (g <= 1.0))


def test_config_from_dict():
    cfg = DisturbanceConfig.from_dict({"mean_speed": 4, "gust_speed": [1, 2]})
    assert cfg.mean_speed == (4.0, 4.0) and cfg.gust_speed == (1.0, 2.0)
    assert DisturbanceConfig.from_dict(None) == DisturbanceConfig()
    with pytest.raises(ValueError, match="unknown"):
        DisturbanceConfig.from_dict({"mean_sped": 4})
    with pytest.raises(ValueError, match="min > max"):
        DisturbanceConfig(mean_speed=(5.0, 1.0))
    with pytest.raises(ValueError, match="non-zero"):
        sample_params(DisturbanceConfig(), 2, seed=0, wall_normal=(0, 0, 0))


def test_mean_heading_range_turns_only_the_mean_wind():
    base = DisturbanceConfig(mean_speed=8.0)
    # The default draws over the whole compass, as before the option existed.
    rng = np.random.default_rng(5)
    speed = rng.uniform(8.0, 8.0, size=6)
    heading = rng.uniform(0.0, 2.0 * math.pi, size=6)
    np.testing.assert_array_equal(sample_params(base, 6, seed=5).mean_wind[:, 0],
                                  speed * np.cos(heading))
    into = sample_params(DisturbanceConfig(mean_speed=8.0, mean_heading_deg=0.0), 6, seed=5)
    np.testing.assert_allclose(into.mean_wind, np.tile([8.0, 0.0, 0.0], (6, 1)), atol=1e-12)
    cone = sample_params(DisturbanceConfig(mean_speed=8.0, mean_heading_deg=(-30, 30)), 64, seed=5)
    toward = np.arctan2(cone.mean_wind[:, 1], cone.mean_wind[:, 0])
    assert (np.abs(toward) <= math.radians(30)).all()
    # Gusts, turbulence and the rest keep the default draw.
    ref = sample_params(base, 6, seed=5)
    for name in ("gust_start", "gust_wind", "turb_amp", "turb_phase", "k_quad", "wall_floor"):
        np.testing.assert_array_equal(getattr(into, name), getattr(ref, name))
