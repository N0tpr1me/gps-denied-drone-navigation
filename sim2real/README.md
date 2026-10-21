# sim2real

Everything that touches PX4 and Gazebo.

Planned: `docker/` (headless PX4 SITL + Gazebo Harmonic image, PX4 release pinned in
docs/decisions.md), `gazebo/` (wall-and-slab world and the wind adapter plugin, see `gazebo/README.md`),
`estimate_bridge/` (GPS-free flight: Gazebo truth through `envs/estimate` into EKF2 as
external vision, see `estimate_bridge/README.md`), `px4/` (the onboard C++ module, its watchdog and the SITL acceptance flight; see `px4/README.md`).
