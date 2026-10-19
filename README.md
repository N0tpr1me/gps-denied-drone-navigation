# GPS-Denied Drone Navigation

Navigation software that lets a small quadrotor fly reliably without GPS (indoors, tunnels, under bridges, confined spaces) using onboard sensors such as a camera and IMU.

Focus: civilian inspection and logistics.

## Approach

1. **Simulation training** — flight control policies trained with reinforcement learning in simulation.
2. **PX4 SITL** — policies validated in PX4 software-in-the-loop.
3. **Real flight** — deployed and flown on a small real drone.

## Status

Early setup. See `CLAUDE.md` for the working rules and `docs/decisions.md` for open choices.
