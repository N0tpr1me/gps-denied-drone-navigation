import numpy as np

from envs.estimate.core import EstimateCorruptionConfig, EstimateCorruptor

OFF = EstimateCorruptionConfig.clean()


def only(**overrides) -> EstimateCorruptionConfig:
    """The clean config with just the named corruption switched on."""
    return EstimateCorruptionConfig(**{**OFF.__dict__, **overrides})


def ramp(steps: int, num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    """A distinct true state per step and env: (steps, N, 3) positions and velocities."""
    k = np.arange(steps)[:, None, None]
    e = np.arange(num_envs)[None, :, None]
    axis = np.array([1.0, -2.0, 0.5])
    return 0.01 * k * axis + e, 0.1 * np.sin(0.01 * k) * axis + 0 * e


def run(model: EstimateCorruptor, pos: np.ndarray, vel: np.ndarray, **reset_kw):
    """Reset on the first sample, step through all, return outputs stacked over steps."""
    model.reset(pos[0], vel[0], **reset_kw)
    outs = [model.step(p, v) for p, v in zip(pos, vel, strict=True)]
    return {f: np.stack([np.asarray(getattr(o, f)) for o in outs]) for f in outs[0]._fields}
