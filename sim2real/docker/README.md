# PX4 SITL image

Headless PX4 SITL with Gazebo Harmonic and the `gz_x500` quadrotor, in one Linux
container. Runs natively on Apple Silicon (arm64) through Docker Desktop and on amd64 (CI,
the GPU server: `docs/gpu-server.md`). It needs no GPU: Gazebo runs server-only and nothing
renders.

- PX4 **v1.17.0**, pinned in `docs/decisions.md`. Moving the pin means re-tuning the
  baseline, so change the `PX4_TAG` build arg, the tag in `run.sh` and the decision row together.
- Other releases, for compatibility checks only (`docs/px4-versions.md`): `PX4_TAG=v1.16.2
  ./run.sh build` builds `structure-hold/px4-sitl:v1.16.2` beside the pinned image, and every
  runner takes it with `--image`. The default everywhere stays v1.17.0, and no benchmark
  number comes from another release.
- MAVSDK-Python 3.17.4 inside the image for scripted flights.
- Nothing graphical by default: Gazebo runs server-only (`HEADLESS=1`). The watch image
  below adds an opt-in browser view.

## Use

Docker Desktop needs at least 4 CPUs and 8 GB of memory (Settings > Resources).

```bash
cd sim2real/docker
./run.sh build        # first build compiles PX4 and pulls Gazebo: 20 to 40 min
./run.sh hover-test   # takeoff to 2.5 m, hover 30 s, land; exits 0 on pass
./run.sh sitl         # just run the simulator (Ctrl-C to stop)
./run.sh shell        # bash in the image
```

`with-sitl <command>` (entrypoint mode) starts SITL and the bridge, runs any command beside
them and copies the ulog to `/out`; `eval/baselines/autotune.py` uses it with the repo's
`entrypoint.sh` mounted, so no rebuild is needed.

`hover-test` writes `runs/<time>_sitl_hover_<commit>/` at the repo root (ignored by git):

- `summary.json`: pass/fail per check, climb and landing times, altitude error and
  horizontal drift over the hover, and Gazebo's real-time factor (mean and min).
- `flight.ulg`: the PX4 flight log, readable with `pyulog`.
- `px4.log`: PX4 console output.

Pass means altitude within 0.5 m of target and horizontal drift under 1 m for the whole
hover, then touchdown and auto-disarm. Thresholds are flags on `hover_test.py`
(`./run.sh hover-test --hover-s 60`, for example).

## Many flights at once (the GPU server)

`python -m eval.baselines.run_baseline ... --jobs auto` flies several flights side by side,
each container pinned to its own CPUs (`--cpus-per-job`, default 4), with its own name and
its own Gazebo partition (`GZ_PARTITION`), so the Gazebo instances cannot discover each
other over Docker's bridge network. Each container already has its own network namespace,
so the MAVLink ports do not clash. The stock runner's rows record Gazebo's real-time factor
(`rtf_min`); fewer jobs if it sags.

## Building in the Claude cloud container

The cloud container has Docker but blocks `packages.osrfoundation.org`, where `ubuntu.sh`
gets Gazebo. `./cloud_build.sh` builds the same image on PX4's published Gazebo Harmonic
dev image instead (`BASE_IMAGE`, `PX4_SETUP=0` build args) and trusts the cloud proxy's CA.
For checking the runner and the amd64 path; benchmark numbers come from `run.sh build`.

## Watching a flight

```bash
./run.sh build-watch   # watch image on top of the SITL image, about a minute
python -m sim2real.px4.run_gazebo wall1m_wind_4ms --watch            # Structure Hold flight
python -m sim2real.estimate_bridge.run_estimate wall1m_wind_4ms --watch   # stock PX4 Hold
```

Then open http://localhost:6080/vnc.html?autoconnect=1&resize=scale. Gazebo's own GUI runs
on a virtual display (Xvfb, CPU rendering) and noVNC serves it; the camera follows the drone
from the side of the wall. Only `--watch` flights use it: they run on
`structure-hold/px4-sitl:v1.17.0-watch` with `WATCH=1` and port 6080 published.

PX4 runs in lockstep with Gazebo, so the flight itself is unchanged; the extra rendering
makes it run slightly slower than real time (on the M4 Mac, real-time factor 0.95 mean and
0.64 min against 0.97 and 0.85 without). Keep `--watch` off for benchmark batches.

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
