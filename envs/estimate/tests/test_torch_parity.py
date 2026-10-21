import numpy as np
import pytest

torch = pytest.importorskip("torch")

from envs.estimate.core import EstimateCorruptionConfig, EstimateCorruptor  # noqa: E402
from envs.estimate.tests.helpers import ramp  # noqa: E402
from envs.estimate.torch_adapter import TorchEstimateCorruptor  # noqa: E402

DT = 0.01
# Every corruption on, at rates high enough that each fires many times in the run.
BUSY = EstimateCorruptionConfig(
    delay_min_s=0.0,
    delay_max_s=0.1,
    bias_walk_std_m_per_sqrt_s=0.2,
    jump_rate_hz=2.0,
    dropout_rate_hz=1.0,
)


def test_numpy_and_torch_agree_to_1e6_on_shared_draws():
    n, steps = 16, 1500
    pos, vel = ramp(steps, n)
    ref = EstimateCorruptor(BUSY, n, DT, seed=11)
    tch = TorchEstimateCorruptor(BUSY, n, DT, seed=0, dtype=torch.float64)
    delays = ref.draw_delay_steps()
    ref.reset(pos[0], vel[0], delay_steps=delays)
    tch.reset(torch.from_numpy(pos[0]), torch.from_numpy(vel[0]), delay_steps=delays)
    fired = {"jumped": 0, "dropout_started": 0}
    for k in range(steps):
        if k == 700:  # a mid-run partial reset must also match
            ref.reset(pos[k], vel[k], env_ids=[0, 5], delay_steps=delays)
            tch.reset(
                torch.from_numpy(pos[k]),
                torch.from_numpy(vel[k]),
                env_ids=[0, 5],
                delay_steps=delays,
            )
        noise = ref.draw_step_noise()
        a = ref.step(pos[k], vel[k], noise)
        b = tch.step(pos[k], vel[k], noise)
        for f in a._fields:
            np.testing.assert_allclose(getattr(b, f).numpy(), getattr(a, f), atol=1e-6, rtol=0)
        fired["jumped"] += int(a.jumped.sum())
        fired["dropout_started"] += int(a.dropout_started.sum())
    assert fired["jumped"] > 100 and fired["dropout_started"] > 50


def test_torch_seeded_runs_identical_float32():
    n, steps = 8, 300
    pos, vel = (torch.from_numpy(x).float() for x in ramp(steps, n))

    def go(seed):
        m = TorchEstimateCorruptor(BUSY, n, DT, seed=seed)
        m.reset(pos[0], vel[0])
        return torch.stack([m.step(p, v).pos for p, v in zip(pos, vel, strict=True)])

    a, b, c = go(3), go(3), go(4)
    assert a.dtype == torch.float32
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_torch_noise_spread_matches_config():
    n = 10_000
    cfg = EstimateCorruptionConfig.clean()
    cfg = EstimateCorruptionConfig(**{**cfg.__dict__, "pos_noise_std_m": 0.05})
    m = TorchEstimateCorruptor(cfg, n, DT, seed=1)
    z = torch.zeros((n, 3))
    m.reset(z, z)
    std = m.step(z, z).pos.std(dim=0)
    assert torch.allclose(std, torch.full((3,), 0.05), rtol=0.05)
