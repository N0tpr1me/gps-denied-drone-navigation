"""The second surface (a deck overhead): absent by default and exactly as before, its
shelter multiplies the wall's, and the deck's edge shedding has the right shape."""

import math

import numpy as np
import pytest

from envs.disturbance import (
    DisturbanceConfig,
    air_velocity,
    disturbance_force,
    sample_params,
    shelter_factor,
    surface2_distance,
    wind_velocity,
)

CALM = dict(num_gusts=0, turb_sigma=0.0)
WALL = dict(wall_point=(0.0, 0.0, 0.0), wall_normal=(1.0, 0.0, 0.0))
# A deck underside 4 m up, free air below it; a bridge along x, so across it is y.
DECK = dict(surface2_point=(0.0, 0.0, 4.0), surface2_normal=(0.0, 0.0, -1.0),
            surface2_across=(0.0, 1.0, 0.0))
FLOW = dict(wall_deflect=(0.2, 0.8), wall_recirc=(0.2, 0.6), wake_ti=(0.1, 0.3),
            shed_amp=(0.1, 0.4))


def test_leaving_it_out_draws_nothing_and_flies_the_same_air():
    cfg = DisturbanceConfig(**FLOW, surface2_shed_amp=(0.1, 0.4))
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    a = sample_params(cfg, 32, rng_a, **WALL)
    b = sample_params(cfg, 32, rng_b, **WALL, surface2_point=None, surface2_normal=None)
    assert rng_a.bit_generator.state == rng_b.bit_generator.state
    assert not a.has_surface2.any()
    pos = np.random.default_rng(4).uniform(0.2, 5.0, (32, 3))
    np.testing.assert_array_equal(disturbance_force(a, 7.5, pos, 0 * pos),
                                  disturbance_force(b, 7.5, pos, 0 * pos))


def test_placing_it_keeps_every_earlier_draw_and_the_air_far_from_it():
    off = sample_params(DisturbanceConfig(**FLOW), 16, seed=4, **WALL)
    on = sample_params(DisturbanceConfig(**FLOW, surface2_shed_amp=(0.1, 0.4)), 16, seed=4,
                       **WALL, **DECK)
    for name in off.__dataclass_fields__:
        if not name.startswith(("surface2", "has_surface2")):
            np.testing.assert_array_equal(getattr(on, name), getattr(off, name), err_msg=name)
    # 200 m below the deck its shelter and shedding have died away.
    pos = np.tile([1.5, 0.0, -196.0], (16, 1))
    np.testing.assert_allclose(air_velocity(on, 3.0, pos), air_velocity(off, 3.0, pos),
                               atol=1e-9)


def test_shelters_multiply():
    cfg = DisturbanceConfig(**CALM, mean_speed=8.0, wall_floor=0.0, surface2_floor=0.0,
                            wall_decay=1.0, surface2_decay=1.0)
    p = sample_params(cfg, 4, seed=1, **WALL, **DECK)
    pos = np.tile([1.0, 0.0, 2.5], (4, 1))  # 1 m from the wall, 1.5 m below the deck
    g_wall, g_deck = 1.0 - math.exp(-1.0), 1.0 - math.exp(-1.5)
    np.testing.assert_allclose(shelter_factor(p, pos), g_wall * g_deck, rtol=1e-12)
    np.testing.assert_allclose(surface2_distance(p, pos), 1.5)
    np.testing.assert_allclose(air_velocity(p, 2.0, pos),
                               g_wall * g_deck * wind_velocity(p, 2.0), rtol=1e-12)


@pytest.mark.parametrize("heading, across", [(90.0, 8.0), (0.0, 0.0), (45.0, 8.0 / 2**0.5)])
def test_deck_shedding_is_driven_by_the_wind_across_it(heading, across):
    cfg = DisturbanceConfig(**CALM, mean_speed=8.0, mean_heading_deg=heading,
                            surface2_floor=1.0, surface2_shed_amp=0.3,
                            surface2_shed_width=2.0, surface2_shed_strouhal=0.12,
                            surface2_wake_length=2.0)
    p = sample_params(cfg, 3, seed=2, **DECK)
    np.testing.assert_allclose(p.surface2_shed_amp, 0.3 * across, atol=1e-12)
    np.testing.assert_allclose(p.surface2_shed_omega, 2 * math.pi * 0.12 * across / 2.0,
                               atol=1e-12)
    # With no shelter left (floor 1) the air is the wind plus the vertical shedding, which
    # decays as exp(-d / wake_length) below the deck.
    pos = np.tile([0.0, 0.0, 2.0], (3, 1))
    t = 1.3
    extra = air_velocity(p, t, pos) - wind_velocity(p, t)
    expected = (p.surface2_shed_amp * np.sin(p.surface2_shed_omega * t + p.surface2_shed_phase)
                * math.exp(-2.0 / 2.0))
    np.testing.assert_allclose(extra[:, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(extra[:, 2], -expected, atol=1e-12)  # along the normal (down)


def test_without_an_across_direction_the_whole_wind_counts():
    cfg = DisturbanceConfig(**CALM, mean_speed=6.0, surface2_shed_amp=0.2)
    deck = dict(DECK, surface2_across=None)
    p = sample_params(cfg, 8, seed=5, **deck)
    np.testing.assert_allclose(p.surface2_shed_amp, 0.2 * 6.0, rtol=1e-12)


def test_config_checks_surface2_lengths():
    with pytest.raises(ValueError, match="surface2"):
        DisturbanceConfig(surface2_shed_width=0.0)
    with pytest.raises(ValueError, match="surface2_normal"):
        sample_params(DisturbanceConfig(), 2, seed=0, surface2_normal=(0.0, 0.0, 0.0))
