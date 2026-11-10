"""Render a Structure Hold flight as a stereo camera + IMU dataset and run a real VIO on it.

The pieces (docs/vio-render.md):

- ``scene``: the structure as textured rectangles (a wall, or a tunnel), with procedural
  concrete textures and an optional blank stretch.
- ``trajectory``: a hold flight in gusts (position, attitude, IMU) at the dataset rates.
- ``camera``: the stereo rig (pinhole, OpenCV axes) on the body and how it sees a point.
- ``raster``: the NumPy renderer (ray cast against the rectangles; image + depth), the
  stand-in for Isaac Sim that runs in CI and in the cloud.
- ``degrade``: fog, dust and low light applied to any rendered image with its depth, the
  same way for both renderers.
- ``euroc``: the EuRoC ASL layout Basalt and ``eval/vio_fit.py`` already read, plus
  Basalt's calibration and config files for the rig.
- ``isaac_backend``: the same scene and cameras in Isaac Sim (server only, untested here).
- ``render``: the run entrypoint (render, run Basalt, fit, compare the conditions).
"""
