# A real VIO on rendered flights: fog, dust and low light

Every robustness number for Structure Hold flies a VIO model (`envs/estimate`) whose numbers
come from papers and from the EuRoC fits (`docs/vio-fit.md`), and the dust, fog and low-light
field fault (`docs/field-faults.md`, `degrade`) made its numbers up: 5 cm noise, 0.2 m/s
velocity noise, a 0.1 m/sqrt(s) walk and a half-second tracking loss every 4 s. Buyers ask
"what VIO did you test with, and what happens in dust?" (Freefly's third deal-breaker,
the GPU plan's item 8). This page is the pipeline that answers it with a real VIO: render
the hold flight as a stereo camera and an IMU would see it, in clear air and in each bad
condition, run Basalt on it, and measure the error the same way the EuRoC flights were
measured.

Two renderers sit behind one interface: a NumPy ray caster that runs anywhere (the cloud
container, CI, the server's CPUs) and Isaac Sim's RTX renderer on the GPU server. Everything
else (the flight, the IMU, the cameras, fog, dust, light, the camera's exposure and noise,
Basalt, the fit) is the same code for both, so the two differ only in how the clean scene
looks.

## The pipeline (`eval/vio_render/`)

| Step | File | What it does |
| --- | --- | --- |
| Flight | `trajectory.py` | 3 s on the floor (the VIO starts there), a 6 s take-off and approach, then a 70 s hold 1.5 m from the wall in 4 m/s blowing into it, with gust steps and turbulence. A point mass with a PID position loop and a lagged acceleration flies it; attitude follows from the acceleration (thrust along body z), so the IMU is exactly consistent with the motion. The IMU (200 Hz) outputs period means, with white noise, bias random walk, a turn-on bias and propeller vibration the VIO is not told about. Integrating the noise-free IMU lands on the truth to millimetres (`test_imu_integrates_back_to_the_truth`) |
| Scene | `scene.py` | A 10 x 5 m concrete wall at x = 0 and a floor (a tunnel is the other scene kind). Procedural concrete: broad stains, mottling, sand grain, pores and pour seams, at about 4 mm a texel, the pixel footprint at the hold |
| Cameras | `camera.py` | Global-shutter stereo pair, 640 x 480, 90 deg wide, 8 cm baseline, 10 cm ahead of the IMU, pitched 10 deg down, 20 Hz |
| Clean frame | `raster.py` / `isaac_backend.py` | Linear radiance and range per pixel. The NumPy renderer: Lambert shading, trilinear mip-mapped textures. Isaac Sim: the same quads and textures on USD meshes, a sun and a sky light, `HdrColor` and `distance_to_camera`; its first frame is checked against the NumPy renderer (range maps within 2 cm, else the run stops) and sets the radiance scale |
| Conditions | `degrade.py` | Fog by Koschmieder's law (contrast falls as `exp(-3.912 range / visibility)`, airlight fills in); dust as grains in 3D around the drone that drift with the wind and settle, seen by both cameras and from frame to frame, drawn as soft discs (out of focus when close), hidden behind the wall, plus the haze of the fine part; low light as a share of a lit hall |
| Camera | `degrade.py` `CameraModel` | Auto exposure up to 20 ms, then gain up to 16x; motion blur from the body's rotation during the exposure; shot noise, read noise, 8-bit output |
| Dataset | `euroc.py` | EuRoC ASL layout (what Basalt and `eval/vio_fit.py` read), Basalt's calibration and config. The calibration Basalt is told is slightly wrong, as on a real drone: focal length 0.3 % off, each camera's mounting 0.1 deg and 1 mm off, camera stamps 2 ms late against the IMU |
| VIO and fit | `render.py` | Basalt (`eval/vio_basalt`, the pinned commit) offline on each dataset, then `eval/vio_fit.py`'s measures per flight, pooled per condition over seeds |

The conditions all come from the same clean frame of the same flight, so a difference
between conditions is the air and the light, never the flight. Everything random is seeded
per seed, condition, frame and camera: a run is reproducible to the bit.

## Running it

```
# Basalt image (once): plain Docker on the server and the Mac; the cloud needs the proxy CA
docker build -t dronenav-basalt eval/vio_basalt          # or eval/vio_basalt/cloud_build.sh
# NumPy renderer, anywhere with Docker: 3 seeds x 9 conditions
python -m training.launch --config eval/configs/vio_render_wall.yaml --device cpu
# Isaac Sim on the GPU server, after the Isaac parity check passed: the whole thing
python -m training.queue training/queues/vio_render.yaml
```

The Isaac run is two steps because Isaac Sim runs inside its own container and Basalt in
another: `eval/vio_render/isaac_render.sh` renders one seed per GPU job and keeps the images
(`vio_render_wall_isaac.yaml`), then `vio_render_pool.yaml` runs Basalt and the
fit on the host with `params.from_run` naming the render runs. The queue does both, waits
for `server_week1`'s `isaac_parity` to have passed, and never uses more than two GPUs.
`eval/tests/test_vio_render.py` checks that the three configs fly the same flights, scene and
conditions.

## First result: NumPy renderer, 3 seeds x 11 conditions (2026-11-09, cloud)

Three hold flights (seeds 1 to 3, 1.5 m from the wall, 4 m/s into it with gusts), each
rendered under eleven conditions, Basalt on every one: runs `20261109-162529`,
`20261109-164347`, `20261109-170144` (one seed each, about 18 min on the 4 cloud cores),
pooled by `20261109-171941_vio_render_pool` (`experiments/LOG.md`). Fit as the EuRoC fits:
the first 10 s (floor and take-off) left out, every window is near hover. Per axis, metres:

| Condition | ATE (mean / worst seed) | Drift 20 s | Drift 60 s | Walk m/sqrt(s) | 60 s cap | Velocity noise m/s |
| --- | --- | --- | --- | --- | --- | --- |
| clear | 0.025 / 0.040 | 0.022 | 0.037 | 0.0036 | 0.07 | 0.007 |
| fog, 20 m visibility | 0.030 / 0.040 | 0.027 | 0.042 | 0.0042 | 0.08 | 0.008 |
| fog, 8 m | 0.028 / 0.034 | 0.025 | 0.029 | 0.0042 | 0.06 | 0.008 |
| fog, 4 m | 0.026 / 0.034 | 0.023 | 0.025 | 0.0041 | 0.05 | 0.010 |
| dust, light (200 grains/m^3, 30 m haze) | 0.027 / 0.035 | 0.024 | 0.028 | 0.0044 | 0.06 | 0.009 |
| dust, medium (600, 15 m) | 0.028 / 0.035 | 0.025 | 0.032 | 0.0041 | 0.07 | 0.010 |
| **dust, heavy (2000, 8 m)** | **0.24 / 0.26** | **0.19** | **0.26** | **0.012** | **0.51** | **0.043** |
| **heavy dust arriving 21 s into the hold** | **0.20 / 0.34** | **0.16** | **0.30** | **0.011** | **0.78** | **0.034** |
| dim (5 % light: 20 ms exposure, gain 3) | 0.031 / 0.044 | 0.028 | 0.040 | 0.0036 | 0.07 | 0.008 |
| dark (1 %: 20 ms, gain 16) | 0.029 / 0.039 | 0.027 | 0.031 | 0.0041 | 0.06 | 0.008 |
| **dark (2 %) with dust (800, 15 m)** | **0.86 / 2.46** | **0.99** | **2.45** | **0.029** | **7.1** | **0.012** |
| *Real Basalt, 10 EuRoC flights (`docs/vio-fit.md`)* | *0.045 to 0.25* | *0.064 (0.015 near hover)* | *0.085* | *0.009 (0.004 near hover)* | *0.19* | *0.017* |
| *`envs/estimate` defaults* | | *0.053* | *0.082* | *0.01* | *0.3* | *0.05* |
| *`degrade` field fault (made up)* | | | | *0.1* | | *0.2, plus a 0.5 s loss every 4 s* |

Basalt never lost tracking, reset or dropped a frame in any of the 33 flights (no gaps, no
jumps): when it fails, it fails silently.

What it says:

- **Clear air is realistic, not perfect.** With the calibration error, time offset and
  vibration in, the clear hold drifts like real Basalt hovering on EuRoC (walk 0.0036 against
  0.004 near hover), with smoother velocity (0.007 against 0.017 m/s). So the renders are a
  fair stand-in for a well-textured wall. They are optimistic on texture: real concrete can
  be blanker, and nothing here is wet, shiny or moving.
- **Fog, darkness and light or medium dust barely matter at a 1.5 m hold.** At most 20 % on
  the error, well inside the `envs/estimate` defaults every policy trains and is scored on.
  Fog takes the far background, and the wall 1.5 m away keeps its contrast; in the dark the
  camera's exposure and gain keep the image usable, at the cost of noise and blur that
  Basalt's patch tracking rides through. The `degrade` field fault's numbers (a 0.1 m/sqrt(s)
  walk, 0.2 m/s velocity noise, a tracking loss every 4 s) are 10 to 25 times rougher than
  any of these measured conditions.
- **Heavy dust is a slow walk, not noise or a loss.** Grains close to the cameras drift with
  the wind and some become tracked features, so Basalt takes part of the air's motion as its
  own: a walk 3 to 4 times faster than clear air, 0.2 to 0.3 m in a minute, up to 0.5 to 0.9 m
  at the worst seed's end, while its velocity still looks fine (0.04 m/s). When the dust
  arrives during a hold the error grows 0.36 to 0.5 m in the 20 s after it arrives on every
  seed. This is the `drift` field fault's shape (a steady walk the estimate itself cannot
  see), not the `degrade` fault's; it is the case the wall guard on the rangefinder is for
  (`docs/field-faults.md`).
- **Starting the VIO in moving dust can ruin it before take-off.** In one of three dark and
  dusty flights Basalt took the drifting grains for motion while the drone sat on the floor
  and ran off at up to 2.8 m/s before take-off, ending the hold 10 m off; in a first look
  (not logged) heavy dust from the start did the same by kilometres. Once airborne it held
  steady again, with a lasting offset. Nothing in its output flags it.

What it does not say yet: how Structure Hold flies on these errors. The numbers above are
the VIO alone, open loop, on a scripted hold flight.

## Next

1. **Isaac Sim renders on the server** (`training/queues/vio_render.yaml`, after the Isaac
   parity check): the same flights and conditions with real shading; the table above is
   the NumPy reference they are compared against.
2. **Fly the measured errors.** Replay each condition's Basalt error into the mock and into
   the Gazebo estimate bridge in place of the modelled VIO, and score v22 against the PD and
   tuned PX4 on them: heavy dust and the dust cloud are the cells that matter.
   The mock half is ready: `eval/configs/fault_rates_dust_measured.yaml` puts the three bold
   rows above into `envs/estimate` terms (the walk, its cap and the velocity noise, with the
   rangefinder spoilt as in `field_degrade`) and scores v22, v24 and the PD on 20,480
   episodes each (`training/queues/server_week3.yaml`). A 128-episode first look in the
   cloud (`experiments/LOG.md`, 2026-11-09 19:49) had no crashes, with v22 and the PD
   following a dust cloud's walk about 0.3 m toward the wall and v24 holding on its range
   input (0.05 m). The Gazebo half is still to do.
3. **Replace the `degrade` fault's made-up numbers** with these (a Proposed row, since the
   field-fault set is a buyer-facing benchmark), and add a dust walk as its own fault.
4. **A pre-take-off check that the VIO is still** (a landed vehicle whose VIO reports
   motion), which would have caught the dust start-up failure; it would live in the module's
   ground-engage checks, so it needs its own decision.
5. **Closed loop**: Basalt fed live from the renderer while Structure Hold flies the
   rendered drone, so the controller's own motion and the VIO's error interact.
