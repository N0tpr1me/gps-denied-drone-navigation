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

RESULTS_PLACEHOLDER
