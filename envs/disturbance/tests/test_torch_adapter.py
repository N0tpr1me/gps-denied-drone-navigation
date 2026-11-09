import numpy as np
import pytest

torch = pytest.importorskip("torch")

from envs.disturbance import DisturbanceConfig, disturbance_force, sample_params  # noqa: E402
from envs.disturbance import torch_adapter as ta  # noqa: E402


def random_state(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 60.0, size=n)
    pos = rng.uniform(-1.0, 6.0, size=(n, 3))
    vel = rng.normal(0.0, 2.0, size=(n, 3))
    return t, pos, vel


NEAR_WALL_FLOW = dict(wall_deflect=(0.3, 0.8), wall_recirc=(0.2, 0.5), wake_ti=(0.1, 0.3),
                      shed_amp=(0.1, 0.4))


@pytest.mark.parametrize("wall, flow, deck", [(False, False, False), (True, False, False),
                                              (True, True, False), (True, True, True)])
def test_numpy_and_torch_agree_to_1e6(wall, flow, deck):
    kwargs = dict(wall_point=(0, 0, 0), wall_normal=(1, 0.3, 0)) if wall else {}
    if deck:
        kwargs.update(surface2_point=(0, 0, 4.0), surface2_normal=(0, 0, -1),
                      surface2_across=(0, 1, 0))
    cfg = DisturbanceConfig(**(NEAR_WALL_FLOW if flow else {}),
                            surface2_shed_amp=(0.1, 0.4) if deck else (0.0, 0.0))
    p = sample_params(cfg, 512, seed=11, **kwargs)
    pt = ta.to_torch(p, dtype=torch.float64)
    for step in range(20):
        t, pos, vel = random_state(512, step)
        expected = disturbance_force(p, t, pos, vel)
        got = ta.disturbance_force(
            pt, torch.from_numpy(t), torch.from_numpy(pos), torch.from_numpy(vel)
        )
        np.testing.assert_allclose(got.numpy(), expected, rtol=0, atol=1e-6)
        # Scalar time too.
        got = ta.disturbance_force(pt, float(t[0]), torch.from_numpy(pos), torch.from_numpy(vel))
        np.testing.assert_allclose(got.numpy(), disturbance_force(p, t[0], pos, vel), atol=1e-6)


def test_float32_is_close_enough_for_training():
    p = sample_params(
        DisturbanceConfig(), 256, seed=12, wall_point=(0, 0, 0), wall_normal=(1, 0, 0)
    )
    pt = ta.to_torch(p)
    t, pos, vel = random_state(256, 99)
    got = ta.disturbance_force(
        pt,
        torch.tensor(t, dtype=torch.float32),
        torch.tensor(pos, dtype=torch.float32),
        torch.tensor(vel, dtype=torch.float32),
    )
    assert got.dtype == torch.float32
    np.testing.assert_allclose(got.numpy(), disturbance_force(p, t, pos, vel), atol=1e-3)


def test_to_torch_keeps_shapes_and_bool_mask():
    p = sample_params(DisturbanceConfig(), 4, seed=13, wall_point=(0, 0, 0), wall_normal=(0, 1, 0))
    pt = ta.to_torch(p, dtype=torch.float64)
    assert pt.has_wall.dtype == torch.bool and bool(pt.has_wall.all())
    assert pt.turb_amp.shape == p.turb_amp.shape and pt.num_envs == 4
    np.testing.assert_allclose(ta.shelter_factor(pt, torch.zeros(4, 3)).numpy(), p.wall_floor)
