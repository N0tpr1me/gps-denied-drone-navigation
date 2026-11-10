"""The surroundings (envs/ambient): air density, battery, rain, rotors near a surface."""

from __future__ import annotations

import numpy as np
import pytest

from envs.ambient import (
    AmbientConfig,
    air_density,
    density_altitude_m,
    hover_headroom,
    nearness,
    sag_gain,
    sample_ambient,
    surface_gain,
)


def test_standard_day_is_the_isa_density():
    assert air_density(15.0, 0.0, 0.0) == pytest.approx(1.225, abs=1e-3)


def test_hot_high_and_humid_air_is_thinner():
    std = air_density(15.0)
    assert air_density(40.0) < std  # hot
    assert air_density(15.0, 2000.0) < std  # high
    assert air_density(30.0, 0.0, 1.0) < air_density(30.0, 0.0, 0.0)  # humid
    assert air_density(-20.0) > std  # cold air is dense
    # Published values: ISA at 2000 m is 1.0066 kg/m^3 (at its own 2 degC), and saturated
    # air at 30 degC is about 1.5 % lighter than dry air at 30 degC.
    assert air_density(15.0 - 0.0065 * 2000.0, 2000.0) == pytest.approx(1.0066, abs=2e-3)
    assert air_density(30.0, 0.0, 1.0) / air_density(30.0) == pytest.approx(0.984, abs=3e-3)
    assert density_altitude_m(air_density(15.0 - 13.0, 2000.0) / 1.225) == pytest.approx(
        2000.0, abs=30.0)


def test_defaults_are_off_and_change_nothing():
    cfg = AmbientConfig()
    assert not cfg.active and not cfg.surfaces and not cfg.air and not cfg.baro_wind
    p = sample_ambient(cfg, 8, seed=0)
    np.testing.assert_allclose(p.power, 1.0, atol=1e-4)
    np.testing.assert_array_equal(p.mass_scale, 1.0)
    np.testing.assert_array_equal(p.battery_sag, 0.0)
    np.testing.assert_array_equal(p.down_force_n, 0.0)
    np.testing.assert_array_equal(p.rotor_radius_m, 0.0)


def test_each_block_switches_active_on():
    for kw in (dict(temperature_c=(30, 40)), dict(altitude_m=1500), dict(humidity=0.9),
               dict(battery_voltage=(0.85, 1.0)), dict(battery_sag=0.1),
               dict(rain_mm_h=(0, 20)), dict(rotor_radius_m=0.2),
               dict(baro_wind_m_per_mps2=0.003)):
        assert AmbientConfig(**kw).active, kw


def test_config_is_validated():
    for kw in (dict(temperature_c=(40, 30)), dict(humidity=1.5), dict(battery_voltage=0.2),
               dict(battery_voltage=1.3),
               dict(battery_sag=0.9), dict(rain_mm_h=-1), dict(rotor_radius_m=2.0),
               dict(ceiling_coef=0.5), dict(unsteady_tau_s=0.0), dict(wall_pull=-0.1)):
        with pytest.raises(ValueError):
            AmbientConfig(**kw)
    with pytest.raises(ValueError, match="unknown ambient keys"):
        AmbientConfig.from_dict({"temprature_c": 20})
    assert AmbientConfig.from_dict({"temperature_c": [0, 30]}).temperature_c == (0.0, 30.0)


def test_sampling_is_seeded_and_in_range():
    cfg = AmbientConfig(temperature_c=(-10, 40), altitude_m=(0, 3000), humidity=(0, 1),
                        battery_voltage=(0.9, 1.14), battery_sag=(0.0, 0.15),
                        rain_mm_h=(0, 50), rotor_radius_m=(0.13, 0.25))
    a, b = sample_ambient(cfg, 500, seed=7), sample_ambient(cfg, 500, seed=7)
    np.testing.assert_array_equal(a.power, b.power)
    assert not np.array_equal(a.power, sample_ambient(cfg, 500, seed=8).power)
    assert a.temperature_c.min() >= -10 and a.temperature_c.max() <= 40
    assert a.density_ratio.min() > 0.6 and a.density_ratio.max() < 1.15
    assert a.battery_voltage.max() > 1.05  # a full pack gives more than nominal
    # Power is density x voltage^2 x what the rain leaves.
    wet = a.rain_mm_h / 50.0
    np.testing.assert_allclose(a.power, a.density_ratio * a.battery_voltage ** 2
                               * (1 - 0.05 * wet))
    np.testing.assert_allclose(a.mass_scale, 1 + 0.02 * wet)


def test_rain_momentum_is_a_few_hundredths_of_a_newton():
    p = sample_ambient(AmbientConfig(rain_mm_h=50.0), 1, seed=0)
    assert 0.01 < p.down_force_n[0] < 0.05


def test_low_battery_at_altitude_on_a_hot_day_leaves_little_headroom():
    p = sample_ambient(AmbientConfig(temperature_c=40, altitude_m=3000, battery_voltage=0.9),
                       1, seed=0)
    # The x500's standard-day thrust over weight is 1.69.
    head = hover_headroom(p.power, p.mass_scale, 1.69)[0]
    assert 0.8 < head < 0.95  # it cannot hover there: a do-not-fly corner, not a test cell


def test_ground_effect_is_cheeseman_bennett():
    r, coef = np.array([0.2]), np.array([1 / 16])
    assert surface_gain(r, np.array([0.2]), coef)[0] == pytest.approx(1 / (1 - 1 / 16))
    assert surface_gain(r, np.array([2.0]), coef)[0] == pytest.approx(1.0, abs=1e-3)
    assert surface_gain(r, np.array([np.inf]), coef)[0] == 1.0  # no surface
    assert surface_gain(np.array([0.0]), np.array([0.3]), coef)[0] == 1.0  # switched off
    # Closer is stronger, and the form stops growing where it stops being trusted.
    d = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
    g = surface_gain(np.full(5, 0.2), d, np.full(5, 1 / 16))
    assert (np.diff(g) <= 0).all() and g[0] == g[1] == pytest.approx(1 / (1 - 4 / 16))
    assert nearness(np.array([0.2]), np.array([-1.0]))[0] == 2.0  # past the surface: capped


def test_sag_is_absorbed_at_hover_and_bites_at_the_top():
    sag, hover = np.array([0.15]), np.array([0.6])
    assert sag_gain(sag, hover, hover)[0] == pytest.approx(1.0)
    assert sag_gain(sag, np.array([1.0]), hover)[0] == pytest.approx(0.85 / 0.91)
    assert sag_gain(sag, np.array([0.2]), hover)[0] > 1.0
    assert sag_gain(np.array([0.0]), np.array([1.0]), hover)[0] == 1.0
