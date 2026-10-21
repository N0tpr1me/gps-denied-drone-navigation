"""Estimate bridge: Gazebo ground truth -> envs.estimate corruption -> PX4 EKF2 external
vision (MAVLink ODOMETRY), so the SITL benchmark flies GPS-free on a VIO-like estimate.

``bridge`` holds the logic (no Gazebo or MAVLink imports, unit-tested); ``run_bridge`` is
the in-container process; ``run_estimate`` flies a scenario from the host and scores it.
"""
