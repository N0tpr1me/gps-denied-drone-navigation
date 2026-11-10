"""EKF2-like fusion (envs/estimate/ekf.py): tracking, dead reckoning, the gate, the delay."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from envs.estimate.ekf import FusionConfig, TorchFusion  # noqa: E402

DT = 0.012


def fly(cfg: FusionConfig, accel_fn, steps: int, fix_fn=None, fresh_fn=None, n: int = 2,
        seed: int = 0):
    """Integrate a known trajectory and fuse its exact fix, as old as the filter assumes;
    returns (true position, true velocity, fused position, fused velocity) per step."""
    f = TorchFusion(cfg, n, DT, seed=seed)
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
        pe, ve = f.step(a, fix_p, fix_v, fresh, true_pos=p)
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


def test_imu_bias_hides_under_the_fix_and_drifts_in_a_dropout():
    # A dropout from 5 s to 8 s while hovering: without the bias the filter dead-reckons on
    # the true acceleration and barely moves; with it the estimate drifts by about
    # bias x t^2 / 2 (0.45 m per 0.1 m/s^2 after 3 s), and the fix had kept it near truth.
    still = lambda t: torch.zeros(1, 3)  # noqa: E731
    gap = lambda t: torch.tensor([not 5.0 <= t < 8.0] * 64)  # noqa: E731
    frozen = {}

    def fix(t, p):
        if not 5.0 <= t < 8.0:
            frozen["p"] = p.clone()
        return frozen["p"]

    def drift(bias: float) -> tuple[float, float]:
        cfg = FusionConfig(enabled=True, imu_noise_mps2=0.0, imu_bias_mps2=bias)
        out = fly(cfg, still, int(8.0 / DT), fix, gap, n=64)
        before = (out[int(5.0 / DT) - 1][2] - out[int(5.0 / DT) - 1][0]).norm(dim=-1).max()
        after = (out[-1][2] - out[-1][0])[:, 0].abs().mean()  # mean |x drift|
        return float(before), float(after)

    assert drift(0.0)[1] < 0.01
    before, after = drift(0.1)
    assert before < 0.01
    # E|N(0, 0.1)| x 3^2 / 2 = 0.08 x 4.5 = 0.36 m, give or take the 64-draw spread
    assert 0.25 < after < 0.5
    with pytest.raises(ValueError):
        FusionConfig(imu_bias_mps2=-0.1)


def test_baro_height_follows_the_barometer_and_never_freezes_or_resets():
    """Barometer height reference (the benchmark since 2026-11-01): the fused height tracks
    the barometer's slow error, not the vision fix, keeps moving through a vision dropout,
    and a restart after it leaves the height where the barometer has it."""
    noise = dict(enabled=True, imu_noise_mps2=0.0, resets=True, baro_noise_m=0.01)
    vision = FusionConfig(**noise)
    baro = FusionConfig(**noise, height_source="baro", baro_walk_m_per_sqrt_s=0.0)
    walk = FusionConfig(**noise, height_source="baro", baro_walk_m_per_sqrt_s=0.05)
    with pytest.raises(ValueError, match="height_source"):
        FusionConfig(height_source="gps")
    with pytest.raises(ValueError, match="true_pos"):
        TorchFusion(baro, 1, DT, 0).step(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3),
                                         torch.ones(1, dtype=torch.bool))
    # Climb at 0.2 m/s; the vision fix reads 0.4 m too high after 1 s and drops out 3 to 4 s.
    climb = lambda t: torch.tensor([[0.0, 0.0, 0.2 / DT]]) if t < DT / 2 else torch.zeros(1, 3)  # noqa: E731
    gap = lambda t: torch.tensor([not 3.0 <= t < 4.0] * 2)  # noqa: E731
    off = lambda t, p: p + torch.tensor([[0.0, 0.0, 0.4]]) if t >= 1.0 else p  # noqa: E731
    steps = int(6.0 / DT)
    v_out = fly(vision, climb, steps, off, gap)
    b_out = fly(baro, climb, steps, off, gap)
    t = torch.arange(steps) * DT
    zerr = lambda out: torch.tensor([float(o[2][0, 2] - o[0][0, 2]) for o in out])  # noqa: E731
    zv, zb = zerr(v_out), zerr(b_out)
    assert zv[t > 4.5].min() > 0.3  # vision height: the restart puts the height on the fix
    assert zb.abs().max() < 0.03  # barometer height: the offset fix and the gap change nothing
    # The barometer's own error walks in: the fused height follows it, in and out of the gap,
    # and the gate never rejects it (EKF2's barometer gate is loose).
    zw = zerr(fly(walk, climb, steps, off, gap, seed=3))
    assert 0.05 < zw.abs().max() < walk.baro_walk_max_m + 0.05
    assert (zw[1:] - zw[:-1]).abs().max() < 0.05  # it walks, it does not step


def test_baro_height_keeps_the_imu_bias_out_of_the_height_in_a_dropout():
    still = lambda t: torch.zeros(1, 3)  # noqa: E731
    gap = lambda t: torch.tensor([not 5.0 <= t < 8.0] * 64)  # noqa: E731
    frozen = {}

    def fix(t, p):
        if not 5.0 <= t < 8.0:
            frozen["p"] = p.clone()
        return frozen["p"]

    cfg = FusionConfig(enabled=True, imu_noise_mps2=0.0, imu_bias_mps2=0.1,
                       height_source="baro", baro_walk_m_per_sqrt_s=0.0)
    out = fly(cfg, still, int(8.0 / DT), fix, gap, n=64)
    err = out[-1][2] - out[-1][0]
    assert err[:, 0].abs().mean() > 0.25  # x drifts under the bias as before
    assert err[:, 2].abs().max() < 0.05  # z stays on the barometer


def test_pressure_events_and_an_outside_offset_move_the_barometer_height():
    """A door opens: the barometer reads the pressure step as height and the fused height
    follows it, while the vehicle has not moved. Off by default, and vision height ignores
    it."""
    still = lambda t: torch.zeros(1, 3)  # noqa: E731
    quiet = dict(enabled=True, imu_noise_mps2=0.0, baro_noise_m=0.005,
                 baro_walk_m_per_sqrt_s=0.0)
    steps = int(60.0 / DT)
    zerr = lambda out: torch.stack([o[2][:, 2] - o[0][:, 2] for o in out])  # noqa: E731
    plain = zerr(fly(FusionConfig(**quiet, height_source="baro"), still, steps, n=64))
    assert float(plain.abs().max()) < 0.02
    door = FusionConfig(**quiet, height_source="baro", baro_step_rate_hz=0.05,
                        baro_step_m=(0.5, 1.0), baro_step_tau_s=1.0)
    assert door.baro_steps and not FusionConfig(**quiet, baro_step_rate_hz=0.05,
                                                baro_step_m=1.0).baro_steps  # vision height
    z = zerr(fly(door, still, steps, n=64, seed=1))
    assert 0.5 < float(z.abs().max()) < 6.0  # a few events in a minute, each 0.5 to 1 m
    assert float((z[1:] - z[:-1]).abs().max()) < 0.05  # reached with its lag, not at once
    assert float(z[-1].abs().mean()) > 0.3  # and it stays
    # With a decay the pressure equalises again.
    leak = FusionConfig(**quiet, height_source="baro", baro_step_rate_hz=0.05,
                        baro_step_m=(0.5, 1.0), baro_step_decay_s=3.0)
    zl = zerr(fly(leak, still, steps, n=64, seed=1))
    assert float(zl[-1].abs().mean()) < 0.5 * float(z[-1].abs().mean())
    # An offset from outside (the wind on the static port) is followed the same way.
    f = TorchFusion(FusionConfig(**quiet, height_source="baro"), 2, DT, seed=0)
    p = torch.zeros(2, 3)
    f.reset(torch.arange(2), p, p)
    off = torch.tensor([0.4, 0.0])
    for _ in range(int(5.0 / DT)):
        pe, _ = f.step(p, p, p, torch.ones(2, dtype=torch.bool), true_pos=p, baro_offset=off)
    assert float(pe[0, 2]) == pytest.approx(0.4, abs=0.03) and abs(float(pe[1, 2])) < 0.03
    for bad in (dict(baro_step_m=(1.0, 0.5)), dict(baro_step_rate_hz=-1.0),
                dict(baro_step_tau_s=0.0), dict(baro_step_decay_s=0.0)):
        with pytest.raises(ValueError):
            FusionConfig(**bad)
    assert FusionConfig.from_dict({"baro_step_m": [0.2, 0.8]}).baro_step_m == (0.2, 0.8)
