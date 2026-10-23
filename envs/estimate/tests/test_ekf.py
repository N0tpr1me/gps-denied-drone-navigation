"""EKF2-like fusion (envs/estimate/ekf.py): tracking, dead reckoning, the gate, the delay."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from envs.estimate.ekf import FusionConfig, TorchFusion  # noqa: E402

DT = 0.012


def fly(cfg: FusionConfig, accel_fn, steps: int, fix_fn=None, fresh_fn=None, n: int = 2):
    """Integrate a known trajectory and fuse its exact fix, as old as the filter assumes;
    returns (true position, true velocity, fused position, fused velocity) per step."""
    f = TorchFusion(cfg, n, DT, seed=0)
    p, v = torch.zeros(n, 3), torch.zeros(n, 3)
    f.reset(torch.arange(n), p, v)
    past = [(p, v)] * (f.delay_steps + 1)
    out = []
    for k in range(steps):
        a = accel_fn(k * DT).expand(n, 3)
        v = v + a * DT
        p = p + v * DT
        past = past[1:] + [(p, v)]
        fix_p, fix_v = past[0]
        fix_p = fix_p if fix_fn is None else fix_fn(k * DT, fix_p)
        fresh = torch.ones(n, dtype=torch.bool) if fresh_fn is None else fresh_fn(k * DT)
        pe, ve = f.step(a, fix_p, fix_v, fresh)
        out.append((p.clone(), v.clone(), pe.clone(), ve.clone()))
    return out


def test_config_defaults_are_off_and_validated():
    assert not FusionConfig().enabled
    with pytest.raises(ValueError):
        FusionConfig(gate_sd=0.0)
    with pytest.raises(ValueError, match="unknown fusion keys"):
        FusionConfig.from_dict({"ev_dealy_s": 0.1})


def test_tracks_an_accelerating_vehicle_with_an_exact_fix():
    cfg = FusionConfig(enabled=True, imu_noise_mps2=0.0)
    acc = lambda t: torch.tensor([[0.5 * torch.sin(torch.tensor(t)).item(), 0.2, 0.0]])  # noqa: E731
    p, v, pe, ve = fly(cfg, acc, 1000)[-1]
    assert torch.allclose(pe, p, atol=0.01) and torch.allclose(ve, v, atol=0.01)


def test_dead_reckons_on_the_imu_while_the_fix_is_stale():
    cfg = FusionConfig(enabled=True, imu_noise_mps2=0.0)
    acc = lambda t: torch.tensor([[0.3, 0.0, -0.1]])  # noqa: E731
    stale_after = lambda t: torch.tensor([t < 2.0, t < 2.0])  # noqa: E731
    # The fix freezes at 2 s (a dropout): the fused estimate keeps following the IMU.
    frozen = {}

    def fix(t, p):
        if t < 2.0:
            frozen["p"] = p.clone()
        return frozen["p"]

    p, v, pe, ve = fly(cfg, acc, 400, fix_fn=fix, fresh_fn=stale_after)[-1]
    assert (p - frozen["p"]).norm(dim=1).min() > 0.5  # the vehicle has moved on
    assert torch.allclose(pe, p, atol=0.02) and torch.allclose(ve, v, atol=0.01)


def test_gate_rejects_a_jump_in_the_fix():
    cfg = FusionConfig(enabled=True, imu_noise_mps2=0.0)
    jump = torch.tensor([[1.0, 0.0, 0.0]])
    out = fly(cfg, lambda t: torch.zeros(1, 3), 200,
              fix_fn=lambda t, p: p + jump if t >= 1.0 else p)
    p, _, pe, _ = out[-1]
    assert torch.allclose(pe, p, atol=0.01)  # a 1 m jump is 10 sd: never fused
    loose = FusionConfig(enabled=True, imu_noise_mps2=0.0, gate_sd=100.0)
    p, _, pe, _ = fly(loose, lambda t: torch.zeros(1, 3), 200,
                      fix_fn=lambda t, p: p + jump if t >= 1.0 else p)[-1]
    assert (pe - p)[:, 0].min() > 0.5  # without the gate the estimate follows it


def test_a_fix_step_is_followed_smoothly():
    # A fix that steps by 5 cm (inside the gate) while hovering: the estimate moves over to it
    # gradually (the velocity fix still says zero), not in one step.
    cfg = FusionConfig(enabled=True, imu_noise_mps2=0.0)
    step = torch.tensor([[0.05, 0.0, 0.0]])
    out = fly(cfg, lambda t: torch.zeros(1, 3), 200,
              fix_fn=lambda t, p: p + step if t >= 0.6 else p)
    x = torch.tensor([float(o[2][0, 0]) for o in out])
    k0 = round(0.6 / DT)
    assert x[:k0].abs().max() < 1e-6
    assert 0.0 < x[k0 + 2] < 0.03  # partway after 24 ms
    assert 0.035 < x[-1] < 0.051  # most of the way after 1.8 s


def test_resets_turn_a_rejected_jump_into_a_step_after_one_second():
    cfg = FusionConfig(enabled=True, imu_noise_mps2=0.0, resets=True)
    jump = torch.tensor([[1.0, 0.0, 0.0]])
    out = fly(cfg, lambda t: torch.zeros(1, 3), 250,
              fix_fn=lambda t, p: p + jump if t >= 1.0 else p)
    x = torch.tensor([float(o[2][0, 0]) for o in out])
    t = torch.arange(len(x)) * DT
    assert x[(t > 1.2) & (t < 1.9)].abs().max() < 0.01  # rejected by the gate at first
    assert x[t > 2.3].min() > 0.95  # then reset onto the fix in one step
    k = int((x > 0.5).float().argmax())
    assert x[k] > 0.95 and 1.9 < float(t[k]) < 2.3  # 1 s after the first rejection, plus delay


def test_resets_restart_fusion_on_the_fix_after_a_dropout():
    cfg = FusionConfig(enabled=True, imu_noise_mps2=0.0, resets=True)
    offset = torch.tensor([[0.0, 0.0, 0.6]])  # the fix comes back 0.6 m off (a VIO re-init)
    gap = lambda t: torch.tensor([not (1.0 <= t < 1.6)] * 2)  # noqa: E731
    out = fly(cfg, lambda t: torch.zeros(1, 3), 200, fresh_fn=gap,
              fix_fn=lambda t, p: p + offset if t >= 1.6 else p)
    z = torch.tensor([float(o[2][0, 2]) for o in out])
    t = torch.arange(len(z)) * DT
    assert z[t < 1.6].abs().max() < 1e-6
    assert z[(t > 1.7) & (t < 1.8)].min() > 0.55  # stepped onto the fix at once
    # Without resets the same fix is rejected (0.6 m is beyond 5 sd).
    out = fly(FusionConfig(enabled=True, imu_noise_mps2=0.0), lambda t: torch.zeros(1, 3), 200,
              fresh_fn=gap, fix_fn=lambda t, p: p + offset if t >= 1.6 else p)
    assert abs(float(out[-1][2][0, 2])) < 0.05


def test_short_gaps_do_not_reset_and_height_waits_five_seconds():
    cfg = FusionConfig(enabled=True, imu_noise_mps2=0.0, resets=True)
    offset = torch.tensor([[0.0, 0.0, 0.6]])
    gap = lambda t: torch.tensor([not (1.0 <= t < 1.3)] * 2)  # noqa: E731, 0.3 s < 0.4 s
    out = fly(cfg, lambda t: torch.zeros(1, 3), 650, fresh_fn=gap,
              fix_fn=lambda t, p: p + offset if t >= 1.3 else p)
    z = torch.tensor([float(o[2][0, 2]) for o in out])
    t = torch.arange(len(z)) * DT
    assert z[t < 6.0].abs().max() < 0.05  # no restart, and the height gate holds for 5 s
    assert z[t > 6.6].min() > 0.55  # then the height resets onto the fix
