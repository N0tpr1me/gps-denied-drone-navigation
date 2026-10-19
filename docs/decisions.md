# Decisions

## Made
- Goal: GPS-denied navigation for small quadrotors, civilian inspection and logistics first.
- Sim and training: Isaac Lab in Docker, PPO to start.
- Flight stack testing: PX4 SITL + ROS 2.
- Every result is compared against a classical baseline.
- Safety layer: the learned policy runs inside a classical safety layer (geofence plus fallback controller) in SITL and on the real drone.

## Open
- Drone framework on Isaac Lab: Aerial Gym or OmniDrones.
- PPO library: rsl_rl, skrl or Stable-Baselines3.
- Real drone: a small PX4 quad, or Crazyflie / Tello (Tello can't run PX4).
- How sessions reach the GPU workstation.
