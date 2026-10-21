# PX4 SITL image

Headless PX4 SITL with Gazebo Harmonic and the `gz_x500` quadrotor, in one Linux
container. Runs natively on Apple Silicon (arm64) through Docker Desktop and on amd64 in CI.

- PX4 **v1.17.0**, pinned in `docs/decisions.md`. Moving the pin means re-tuning the
  baseline, so change the `PX4_TAG` build arg, the tag in `run.sh` and the decision row together.
- MAVSDK-Python 3.17.4 inside the image for scripted flights.
- Nothing graphical: Gazebo runs server-only (`HEADLESS=1`).

## Use

Docker Desktop needs at least 4 CPUs and 8 GB of memory (Settings > Resources).

```bash
cd sim2real/docker
./run.sh build        # first build compiles PX4 and pulls Gazebo: 20 to 40 min
./run.sh hover-test   # takeoff to 2.5 m, hover 30 s, land; exits 0 on pass
./run.sh sitl         # just run the simulator (Ctrl-C to stop)
./run.sh shell        # bash in the image
```

`hover-test` writes `runs/<time>_sitl_hover_<commit>/` at the repo root (ignored by git):

- `summary.json`: pass/fail per check, climb and landing times, altitude error and
  horizontal drift over the hover, and Gazebo's real-time factor (mean and min).
- `flight.ulg`: the PX4 flight log, readable with `pyulog`.
- `px4.log`: PX4 console output.

Pass means altitude within 0.5 m of target and horizontal drift under 1 m for the whole
hover, then touchdown and auto-disarm. Thresholds are flags on `hover_test.py`
(`./run.sh hover-test --hover-s 60`, for example).

## Talking to the simulator

MAVLink clients run inside the container (`docker exec -it px4-sitl bash`), where PX4
sends API traffic to UDP 14540 and ground-station traffic to 14550. Reaching it from the
Mac itself needs extra MAVLink forwarding, which later tasks add when they need it.

## Notes

- GPS is on, as in stock SITL. `sim2real/estimate_bridge` turns it off and feeds EKF2 a
  corrupted external-vision estimate (`hover-test --local-position-only`, bridge started by
  the entrypoint when `ESTIMATE_BRIDGE_PROFILE` is set).
- `gz_x500` flies in Gazebo's `default` world; the wall-and-slab world arrives with the
  Gazebo tasks in `sim2real/gazebo/`.
- The image is several GB. Remove it with `docker image rm structure-hold/px4-sitl:v1.17.0`.
