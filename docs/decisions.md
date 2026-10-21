# Decisions

One row per decision. Change a row only in its own PR, and say why in the PR.
Full reasoning: the Structure Hold build plan and execution plan (project docs, 21 Oct 2026).

## Product and scope

| Decision | Status | Date | Why / what would change it |
| --- | --- | --- | --- |
| Build one onboard PX4 flight mode, "Structure Hold": hold and 5 m passes 1 to 3 m from a structure in gusts, no GPS, robust to bad position estimates | Decided | 2026-10-21 | Product strategy doc |
| Out of scope: VIO/SLAM, mapping, planning, obstacle avoidance, ground station, vision in the policy | Decided | 2026-10-21 | Use the drone's existing position source; accept position + covariance |
| Classical watchdog (no learning) hands back to stock PX4 on any trip | Decided | 2026-10-21 | Non-negotiable for buyers' safety review |
| Every result compared against tuned PX4 (`eval/baselines/px4_tuned_v1.params`), measured before any RL | Decided | 2026-10-21 | The benchmark is the asset |

## Technical

| Decision | Status | Date | Why / what would change it |
| --- | --- | --- | --- |
| Learned outer loop: body rates + collective thrust at 100 Hz over PX4's stock rate controller; MLP ~10k params, 3-step history, asymmetric critic | Decided | 2026-10-21 | Rate interface is the bottleneck at Gate 2 → try direct motor commands |
| RL library: rsl_rl PPO, version pinned in `pyproject.toml` when training code lands | Decided | 2026-10-21 | Only if rsl_rl lacks something needed |
| Simulator: Isaac Lab native quadrotor task (direct workflow), not Aerial Gym | **Recommended, awaiting Zain** | 2026-10-21 | Isaac Lab not training the stock task within 5 working days on the server → batched PyTorch sim |
| PX4 integration: onboard C++ module from day one, built into `px4_sitl` first | **Recommended, awaiting Zain** | 2026-10-21 | Internal module cannot register as a mode → small commander patch |
| Test airframe: ModalAI Starling 2; fallback Holybro X500 V2 + Pixhawk 6C | **Recommended, awaiting Zain** | 2026-10-21 | ModalAI build flow won't take the module within 2 weeks |
| Develop against a batched PyTorch quadrotor ("mock", `envs/sims/torch_quad.py`) until the GPU server exists; Isaac Lab plugs in behind the same seam (thrust, body torque, external force in; `QuadState` out) | Decided | 2026-10-21 | GPU server access comes later; specs unknown |
| PX4 release pinned at Phase 0 (not Phase 2) and held through Gate 3; tag recorded here by the SITL image task | Decided | 2026-10-21 | A baseline tuned on one release is not fair on another. Moving the pin means re-tuning the baseline |
| PX4 pinned tag: **v1.17.0** (`sim2real/docker`: Ubuntu 24.04, Gazebo Harmonic, MAVSDK-Python 3.17.4) | Decided | 2026-10-21 | Latest stable release at pin time; builds from source on arm64 and amd64. Moving it means re-tuning the baseline |
| Shared core code is NumPy + PyYAML only, Python 3.10 to 3.12; PyTorch only in the batched sim and training | Decided | 2026-10-21 | Must import in Isaac Lab, the SITL image and the Mac |
| One run on one GPU at a time; `training/launch.py` gates on `nvidia-smi`, refuses >1 GPU or >1 h without `--approved` | Decided | 2026-10-21 | Compute plan; project rules |

## Ideas parked (scope creep goes here, not into code)

| Idea | Date | Note |
| --- | --- | --- |
