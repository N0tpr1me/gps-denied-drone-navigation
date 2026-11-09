"""A deck over the hold (cfg.deck_clearance; docs/bridge-holds.md): off by default with the
same episodes as before, placed per episode, in the wind, and a contact like the wall."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from envs.disturbance import DisturbanceConfig  # noqa: E402
from envs.estimate import EstimateCorruptionConfig  # noqa: E402
from envs.structure_hold import NUM_ACTIONS, StructureHoldConfig, StructureHoldEnv  # noqa: E402

CALM = DisturbanceConfig(mean_speed=(0.0, 0.0), num_gusts=0, turb_sigma=(0.0, 0.0))


def hover(env: StructureHoldEnv) -> torch.Tensor:
    a = torch.zeros(env.num_envs, NUM_ACTIONS)
    a[:, 3] = 2.0 * env.cfg.quad.hover_thrust / env.cfg.quad.max_thrust - 1.0
    return a


def test_no_deck_by_default_and_the_same_episodes():
    a = StructureHoldEnv(StructureHoldConfig(), num_envs=16, seed=3)
    b = StructureHoldEnv(StructureHoldConfig(deck_clearance=None), num_envs=16, seed=3)
    a.reset(), b.reset()
    assert torch.isinf(a.deck_z).all() and not a.wind.has_surface2.any()
    torch.testing.assert_close(a.state.pos, b.state.pos)
    torch.testing.assert_close(a.wind.mean_wind, b.wind.mean_wind)


def test_deck_sits_its_clearance_over_the_hold_and_shelters_the_wind():
    cfg = StructureHoldConfig(deck_clearance=(1.0, 2.0),
                              disturbance=DisturbanceConfig(surface2_shed_amp=(0.1, 0.4)))
    env = StructureHoldEnv(cfg, num_envs=64, seed=4)
    env.reset()
    gap = env.deck_z - env.hold_point[:, 2]
    assert bool(((gap >= 1.0) & (gap <= 2.0)).all())
    assert bool(env.wind.has_surface2.all())
    torch.testing.assert_close(env.wind.surface2_point[:, 2], env.deck_z)
    # Never starts within reach of the deck.
    assert bool((env.deck_clearance() > cfg.contact_radius_m).all())


def test_touching_the_deck_is_a_contact():
    cfg = StructureHoldConfig(deck_clearance=(0.6, 0.6), disturbance=CALM,
                              estimate=EstimateCorruptionConfig.clean(),
                              start_offset_std_m=0.0, start_yaw_std_rad=0.0,
                              episode_length_s=3.0)
    env = StructureHoldEnv(cfg, num_envs=2, seed=0)
    env.reset()
    a = hover(env)
    a[:, 3] = 1.0  # full thrust: straight up into the deck
    for _ in range(int(3.0 / cfg.dt)):
        _, _, done, _ = env.step(a)
        if bool(done.all()):
            break
    assert env.counts["contact"] == 2


def test_config_checks_deck_clearance():
    with pytest.raises(ValueError, match="deck_clearance"):
        StructureHoldConfig(deck_clearance=(0.3, 1.0))
    cfg = StructureHoldConfig.from_dict({"deck_clearance": 1.5})
    assert cfg.deck_clearance == (1.5, 1.5)
    assert StructureHoldConfig.from_dict({"deck_clearance": None}).deck_clearance is None
    assert math.isinf(float(StructureHoldEnv(StructureHoldConfig(), 1).deck_z[0]))
