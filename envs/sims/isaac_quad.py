"""Isaac Lab backend behind the simulator seam (``make_sim("isaac", ...)``).

Same calls and the same ``QuadState`` as the mock (``torch_quad.py``):

    sim = IsaacQuadSim(QuadParams(), num_envs=4096, dt=0.01, device="cuda")
    sim.reset()                                     # all envs level at the origin
    sim.set_airframe(env_ids, mass=..., inertia=...)   # optional, per env (randomisation)
    state = sim.step(thrust, torque_body, force_ext_world)   # [, torque_ext_body]

How it maps onto Isaac Lab (pinned to Isaac Lab 2.3.2 on Isaac Sim 5.1.0, docs/isaac-lab.md):

- One PhysX rigid body per env, spawned with Isaac Lab's ``InteractiveScene`` and driven the
  way Isaac Lab's direct-workflow quadcopter task (``Isaac-Quadcopter-Direct-v0``) drives its
  Crazyflie: a body-frame force and torque through ``permanent_wrench_composer`` every
  physics step. The task's Crazyflie articulation is not used: its four propeller links add
  mass, inertia and spin that the mock does not have, and the seam's contract is "a backend
  moves one rigid body". Mass and diagonal inertia are written into PhysX from
  ``QuadParams`` (or, per env, from ``set_airframe``) and read back; a mismatch fails loudly.
- Everything the mock adds on top of rigid-body physics is done here in torch with the
  mock's own formulas, so the two backends share it exactly: thrust clipped to
  ``[0, max_thrust]`` and torque to ``±max_torque``, then the first-order motor lag
  ``motor_tau`` per physics sub-step, all per env as ``set_airframe`` leaves them; then the
  optional external body torque (an aerodynamic moment, neither clipped nor lagged).
  Physics runs at ``dt / substeps``, as the mock does.
- No drag, no damping, no sleeping, no collisions, no ground: PhysX's default angular
  damping (0.05) and sleep thresholds are switched off, gravity is the mock's 9.80665 m/s^2,
  gyroscopic forces are on (the mock integrates Euler's equation with the gyro term).
- The rate loop (``RatePid``), the PX4-like rate lag (``rate_lag_s``) and the action limit
  (``max_rate_rad_s``) live in ``envs/structure_hold``, above the seam, and so are identical
  on both backends by construction.
- Frames: Isaac Lab is Z-up world with (w, x, y, z) quaternions, as our ENU / FLU
  convention; positions are returned relative to each env's origin; PhysX keeps angular
  velocity in the world frame, so it is rotated to body FLU on the way out and back to
  world on reset.

Isaac Lab is imported lazily, inside ``_PhysxWorld``: importing this module needs only
torch, so CI and the Mac never need Isaac. Isaac Lab runs one physics scene per process,
so building a second ``IsaacQuadSim`` closes the first (stepping the old one then fails
loudly). Training and scoring build their envs one after another, so this never bites.
"""

from __future__ import annotations

import atexit
import importlib.util
from typing import Any

import numpy as np
import torch

from envs.sims.quad_state import QuadState
from envs.sims.torch_quad import GRAVITY, QuadParams, quat_rotate, quat_rotate_inverse

ISAACLAB_VERSION = "2.3.2"
ISAACSIM_VERSION = "5.1.0"
DOCKER_IMAGE = f"nvcr.io/nvidia/isaac-lab:{ISAACLAB_VERSION}"
ENV_SPACING_M = 2.0  # where Isaac Lab places env origins; collisions are off, so any value works
BODY_SIZE_M = (0.30, 0.26, 0.06)  # the cuboid's look only: mass and inertia are set explicitly

# PhysX rigid-body settings (Isaac Lab ``RigidBodyPropertiesCfg`` field names). Damping,
# sleeping and velocity caps are PhysX features the mock does not have, so they are off.
RIGID_BODY_PROPS: dict[str, Any] = {
    "rigid_body_enabled": True,
    "kinematic_enabled": False,
    "disable_gravity": False,
    "linear_damping": 0.0,  # PhysX default 0
    "angular_damping": 0.0,  # PhysX default 0.05 would slow every rotation
    "max_linear_velocity": 1000.0,  # m/s
    "max_angular_velocity": 1.0e5,  # deg/s in Isaac Lab's units; PhysX default ~100 rad/s
    "enable_gyroscopic_forces": True,
    "retain_accelerations": False,
    "sleep_threshold": 0.0,  # a hovering body is slow; it must never be put to sleep
    "stabilization_threshold": 0.0,
    "solver_position_iteration_count": 4,  # no contacts, so iterations do not matter
    "solver_velocity_iteration_count": 0,
}


class IsaacLabUnavailable(ImportError):
    """Isaac Lab is not installed in this Python; see docs/isaac-lab.md."""


def isaac_available() -> bool:
    """True when Isaac Lab and Isaac Sim import in this Python (no app is started)."""
    return all(importlib.util.find_spec(m) is not None for m in ("isaaclab", "isaacsim"))


def parameter_map(params: QuadParams, dt: float) -> list[tuple[str, Any, str]]:
    """How each mock number reaches Isaac Lab: (QuadParams field, value, where it goes)."""
    return [
        ("mass", params.mass, "PhysX rigid body mass (set_masses, read back)"),
        ("inertia", params.inertia, "PhysX diagonal inertia, principal axes = body FLU "
         "(set_inertias + set_coms, read back)"),
        ("max_thrust", params.max_thrust, "per-env clip in IsaacQuadSim.step, as the mock"),
        ("max_torque", params.max_torque, "per-env clip in IsaacQuadSim.step, as the mock"),
        ("motor_tau", params.motor_tau, "per-env first-order lag in torch per physics "
         "sub-step, as the mock"),
        ("substeps", params.substeps, f"PhysX dt = {dt} / {params.substeps} = "
         f"{dt / params.substeps:g} s, one wrench write per sub-step"),
        ("(gravity)", GRAVITY, "SimulationCfg.gravity = (0, 0, -g)"),
        ("(drag, damping)", 0.0, "linear_damping = angular_damping = 0; the wind force "
         "carries the drone's own drag"),
    ]


class IsaacQuadSim:
    """``num_envs`` quadrotors as PhysX rigid bodies in Isaac Lab. Same API as TorchQuadSim.

    ``world`` is for tests only: an object with the ``_PhysxWorld`` methods (``origins``,
    ``set_mass_properties``, ``write_state``, ``step``, ``read``, ``closed``) that stands in
    for PhysX, so the glue in this class (frames, clipping, motor lag, per-env airframes,
    resets) is tested without Isaac Lab.
    """

    backend = "isaac"

    def __init__(
        self,
        params: QuadParams,
        num_envs: int,
        dt: float,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        world: Any = None,
    ):
        if num_envs < 1 or dt <= 0:
            raise ValueError("num_envs must be >= 1 and dt > 0")
        if dtype != torch.float32:
            raise ValueError("PhysX state is float32; the Isaac backend supports only float32")
        self.params = params
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", 0)
        self.dtype = dtype
        self.h = self.dt / params.substeps
        if world is None:
            if not isaac_available():
                raise IsaacLabUnavailable(
                    "the isaac backend needs Isaac Lab "
                    f"{ISAACLAB_VERSION} (Isaac Sim {ISAACSIM_VERSION}) in this Python; run "
                    f"inside {DOCKER_IMAGE} as docs/isaac-lab.md describes, or use --sim torch"
                )
            world = _PhysxWorld(params, self.num_envs, self.h, str(self.device))
        self.world = world
        # Per-env airframe, (N,) or (N, 3), with the mock's names; every env starts as params.
        n, dev = self.num_envs, self.device
        self.inertia = torch.zeros(n, 3, dtype=dtype, device=dev)
        self.max_torque = torch.zeros(n, 3, dtype=dtype, device=dev)
        self.inv_mass = torch.zeros(n, dtype=dtype, device=dev)
        self.hover_thrust = torch.zeros(n, dtype=dtype, device=dev)
        self.max_thrust = torch.zeros(n, dtype=dtype, device=dev)
        self.lag = torch.zeros(n, dtype=dtype, device=dev)
        self.set_airframe()
        self._state = QuadState.hover(self.num_envs, device=self.device, dtype=dtype)
        self.thrust = torch.zeros(self.num_envs, dtype=dtype, device=self.device)
        self.torque = torch.zeros(self.num_envs, 3, dtype=dtype, device=self.device)
        self.reset()

    @property
    def state(self) -> QuadState:
        return self._state

    def _ids(self, env_ids: Any) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device)
        if ids.dtype == torch.bool:
            ids = ids.nonzero(as_tuple=False).squeeze(-1)
        return ids.long()

    def set_airframe(self, env_ids: Any = None, *, mass: Any = None, inertia: Any = None,
                     max_thrust: Any = None, max_torque: Any = None,
                     motor_tau: Any = None) -> None:
        """Same call as ``TorchQuadSim.set_airframe``: ``env_ids`` (default all) fly another
        airframe from the next step on; each argument is a scalar or one value (row) per
        listed env, and one left out returns to ``QuadParams``. Mass and inertia go into
        PhysX (read back); limits and motor lag stay here, as in the mock."""
        self._check_open()
        ids = self._ids(env_ids)
        n = int(ids.numel())
        if n == 0:
            return
        p = self.params

        def rows(value: Any, default: Any, width: int | None) -> np.ndarray:
            v = np.asarray(default if value is None else value, dtype=np.float64)
            shape = (n,) if width is None else (n, width)
            return np.broadcast_to(v, shape).copy()

        m = rows(mass, p.mass, None)
        inertia_ = rows(inertia, p.inertia, 3)
        thrust = rows(max_thrust, p.max_thrust, None)
        torque = rows(max_torque, p.max_torque, 3)
        tau = rows(motor_tau, p.motor_tau, None)
        if (m <= 0).any() or (inertia_ <= 0).any() or (thrust <= 0).any() or (torque <= 0).any():
            raise ValueError("mass, inertia, max_thrust and max_torque must be positive")
        if (tau < 0).any():
            raise ValueError("motor_tau must be >= 0")
        # Derived in float64 and cast once, exactly as the mock does.
        lag = np.where(tau > 0, 1.0 - np.exp(-self.h / np.where(tau > 0, tau, 1.0)), 1.0)
        t = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)  # noqa: E731
        self.world.set_mass_properties(ids, t(m), t(inertia_))
        self.inv_mass[ids] = t(1.0 / m)
        self.hover_thrust[ids] = t(m * GRAVITY)
        self.inertia[ids] = t(inertia_)
        self.max_thrust[ids] = t(thrust)
        self.max_torque[ids] = t(torque)
        self.lag[ids] = t(lag)

    def _check_open(self) -> None:
        if self.world.closed:
            raise RuntimeError(
                "this Isaac sim was replaced by a newer one: Isaac Lab runs one physics scene "
                "per process, so only the most recently built IsaacQuadSim can step"
            )

    def reset(self, env_ids: Any = None, state: QuadState | None = None) -> None:
        """Same semantics as ``TorchQuadSim.reset``: rows for ``env_ids`` or a full batch,
        ``None`` = level and at rest at the origin; motors restart at hover thrust."""
        self._check_open()
        ids = self._ids(env_ids)
        n = int(ids.numel())
        if n == 0:
            return
        if state is None:
            new = QuadState.hover(n, device=self.device, dtype=self.dtype)
        else:
            new = QuadState(*(torch.as_tensor(x, dtype=self.dtype, device=self.device)
                              for x in state))
            if new.num_envs == self.num_envs and n != self.num_envs:
                new = QuadState(*(x[ids] for x in new))
            if new.num_envs != n:
                raise ValueError(f"state has {new.num_envs} rows, expected {n} or {self.num_envs}")
            if not bool(new.finite().all()):
                raise ValueError("reset state must be finite")
        q = new.quat / new.quat.norm(dim=-1, keepdim=True)
        self.world.write_state(ids, new.pos, q, new.vel, quat_rotate(q, new.ang_vel))
        self._state.pos[ids] = new.pos
        self._state.quat[ids] = q
        self._state.vel[ids] = new.vel
        self._state.ang_vel[ids] = new.ang_vel
        self.thrust[ids] = self.hover_thrust[ids]
        self.torque[ids] = 0.0

    def step(self, thrust: Any, torque_body: Any, force_ext_world: Any,
             torque_ext_body: Any = None) -> QuadState:
        """Advance every env by ``dt``; shapes and units as ``TorchQuadSim.step``, including
        the optional ``(N, 3)`` external body torque added after the mixer and the lag."""
        self._check_open()
        p = self.params
        thrust_cmd = torch.as_tensor(thrust, dtype=self.dtype, device=self.device)
        torque_cmd = torch.as_tensor(torque_body, dtype=self.dtype, device=self.device)
        f_ext = torch.as_tensor(force_ext_world, dtype=self.dtype, device=self.device)
        tau_ext = (None if torque_ext_body is None else
                   torch.as_tensor(torque_ext_body, dtype=self.dtype, device=self.device))
        if thrust_cmd.shape != (self.num_envs,):
            raise ValueError(f"thrust must be ({self.num_envs},), got {tuple(thrust_cmd.shape)}")
        for name, x in (("torque_body", torque_cmd), ("force_ext_world", f_ext),
                        ("torque_ext_body", tau_ext)):
            if x is not None and x.shape != (self.num_envs, 3):
                raise ValueError(f"{name} must be ({self.num_envs}, 3), got {tuple(x.shape)}")
        thrust_cmd = torch.minimum(thrust_cmd.clamp(min=0.0), self.max_thrust)
        torque_cmd = torch.maximum(torch.minimum(torque_cmd, self.max_torque), -self.max_torque)

        lag, lag3 = self.lag, self.lag[:, None]
        for _ in range(p.substeps):
            self.thrust = self.thrust + lag * (thrust_cmd - self.thrust)
            self.torque = self.torque + lag3 * (torque_cmd - self.torque)
            torque = self.torque if tau_ext is None else self.torque + tau_ext
            force_b, torque_b = body_wrench(self._state.quat, self.thrust, torque, f_ext)
            self.world.step(force_b, torque_b)
            self._state = QuadState(*self.world.read())
        return self._state

    def close(self) -> None:
        self.world.close()


def body_wrench(quat: torch.Tensor, thrust: torch.Tensor, torque: torch.Tensor,
                force_ext_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The force and torque to hand PhysX in the body (link) frame for one sub-step: thrust
    along body +z plus the world external force rotated into the body at the sub-step's
    starting attitude, which is the attitude the mock uses for the same sub-step."""
    force_b = quat_rotate_inverse(quat, force_ext_world)
    force_b = force_b + torch.stack(
        [torch.zeros_like(thrust), torch.zeros_like(thrust), thrust], -1)
    return force_b, torque


# ---------------------------------------------------------------- the only Isaac Lab code

_APP: Any = None
_ACTIVE: _PhysxWorld | None = None


def _launch_app(device: str) -> None:
    """Start Isaac Sim headless once per process (Isaac Lab modules import only after it)."""
    global _APP
    if _APP is not None:
        return
    try:  # already inside a running Kit app (for example a script started by isaaclab.sh)
        import omni.kit.app  # type: ignore[import-not-found]

        if omni.kit.app.get_app().is_running():
            _APP = "external"
            return
    except ImportError:
        pass
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    _APP = AppLauncher(headless=True, device=device).app
    atexit.register(_APP.close)


class _PhysxWorld:
    """``num_envs`` PhysX rigid bodies in an Isaac Lab scene, stepped at ``physics_dt``."""

    def __init__(self, params: QuadParams, num_envs: int, physics_dt: float, device: str):
        global _ACTIVE
        # The app itself always wants the GPU; physics may still run on PhysX's CPU pipeline.
        _launch_app("cuda:0" if torch.cuda.is_available() else device)
        import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
        from isaaclab.assets import RigidObjectCfg  # type: ignore[import-not-found]
        from isaaclab.scene import (  # type: ignore[import-not-found]
            InteractiveScene,
            InteractiveSceneCfg,
        )
        from isaaclab.utils import configclass  # type: ignore[import-not-found]

        if _ACTIVE is not None:
            _ACTIVE.close()
        self.closed = False
        self.physics_dt = float(physics_dt)
        self.num_envs = num_envs
        sim_utils.create_new_stage()
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.physics_dt, device=device, gravity=(0.0, 0.0, -GRAVITY),
        ))
        body_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Quad",
            spawn=sim_utils.CuboidCfg(
                size=BODY_SIZE_M,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(**RIGID_BODY_PROPS),
                mass_props=sim_utils.MassPropertiesCfg(mass=params.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

        @configclass
        class _SceneCfg(InteractiveSceneCfg):
            quad: RigidObjectCfg = body_cfg

        self.scene = InteractiveScene(_SceneCfg(num_envs=num_envs, env_spacing=ENV_SPACING_M,
                                                replicate_physics=True))
        self.sim.reset()
        self.body = self.scene["quad"]
        self.origins = self.scene.env_origins.clone()
        # Centre of mass at the body origin, principal axes = body FLU, for every env. PhysX
        # lays a COM pose out as (x, y, z, qx, qy, qz, qw).
        view = self.body.root_physx_view
        shape = view.get_coms().shape
        coms = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).repeat(num_envs, 1)
        view.set_coms(coms.reshape(shape), torch.arange(num_envs, dtype=torch.int32))
        everyone = torch.arange(num_envs)
        self.set_mass_properties(everyone, torch.full((num_envs,), params.mass),
                                 torch.tensor(params.inertia).expand(num_envs, 3))
        _ACTIVE = self

    def set_mass_properties(self, ids: torch.Tensor, mass: torch.Tensor,
                            inertia: torch.Tensor) -> None:
        """Mass (n,) and diagonal inertia (n, 3) for envs ``ids``, then read back. PhysX's
        tensor API takes full-size CPU tensors plus int32 indices of the rows to apply."""
        view = self.body.root_physx_view
        idx = ids.detach().to("cpu", torch.int32)
        rows = idx.long()
        current = view.get_masses()
        masses = current.clone().reshape(self.num_envs)
        masses[rows] = mass.detach().cpu().to(masses.dtype)
        view.set_masses(masses.reshape(current.shape), idx)
        current = view.get_inertias()
        flat = current.clone().reshape(self.num_envs, 9)
        flat[rows] = 0.0
        diag = inertia.detach().cpu().to(flat.dtype)
        flat[rows, 0], flat[rows, 4], flat[rows, 8] = diag[:, 0], diag[:, 1], diag[:, 2]
        view.set_inertias(flat.reshape(current.shape), idx)
        got_m = view.get_masses().reshape(self.num_envs)[rows]
        got_i = view.get_inertias().reshape(self.num_envs, 9)[rows][:, [0, 4, 8]]
        want_m = mass.detach().cpu().to(got_m.dtype)
        if not (torch.allclose(got_m, want_m, rtol=1e-5)
                and torch.allclose(got_i, diag.to(got_i.dtype), rtol=1e-5)):
            raise RuntimeError(
                f"PhysX did not take the mass properties: mass {got_m[:3].tolist()} (want "
                f"{want_m[:3].tolist()}), inertia {got_i[0].tolist()} (want {diag[0].tolist()})"
            )

    def write_state(self, ids: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor,
                    vel_w: torch.Tensor, ang_vel_w: torch.Tensor) -> None:
        self.body.write_root_pose_to_sim(torch.cat([pos + self.origins[ids], quat], -1),
                                         env_ids=ids)
        self.body.write_root_velocity_to_sim(torch.cat([vel_w, ang_vel_w], -1), env_ids=ids)
        self.body.reset(ids)  # clears any wrench left in the composers for those envs

    def step(self, force_b: torch.Tensor, torque_b: torch.Tensor) -> None:
        forces, torques = force_b[:, None, :], torque_b[:, None, :]
        composer = getattr(self.body, "permanent_wrench_composer", None)
        if composer is not None:  # Isaac Lab >= 2.3
            composer.set_forces_and_torques(forces=forces, torques=torques)
        else:  # older Isaac Lab: same effect, body frame
            self.body.set_external_force_and_torque(forces, torques)
        self.body.write_data_to_sim()
        self.sim.step(render=False)
        self.body.update(self.physics_dt)

    def read(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Position relative to the env origin (ENU), quaternion (w, x, y, z), world
        velocity, body angular velocity; cloned, since Isaac Lab updates its buffers in place."""
        d = self.body.data
        return ((d.root_pos_w - self.origins).clone(), d.root_quat_w.clone(),
                d.root_lin_vel_w.clone(), d.root_ang_vel_b.clone())

    def close(self) -> None:
        global _ACTIVE
        if self.closed:
            return
        self.closed = True
        sim = self.sim
        handle = getattr(sim, "_app_control_on_stop_handle", None)
        if handle is not None:  # headless Isaac Lab would otherwise shut the app on stop
            handle.unsubscribe()
            sim._app_control_on_stop_handle = None
        sim.stop()
        sim.clear_all_callbacks()
        sim.clear_instance()
        if _ACTIVE is self:
            _ACTIVE = None
