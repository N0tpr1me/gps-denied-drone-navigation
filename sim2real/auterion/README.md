# Structure Hold as an Auterion app

The same Structure Hold flight logic as the onboard module (`../px4/structure_hold`), packaged
as a PX4 external flight mode that runs as an AuterionOS app on Skynode, or next to stock
PX4 on any companion computer. Why and how: `docs/auterion-app.md`.

| Path | What |
| --- | --- |
| `structure_hold_app/include/structure_hold_app/HoldCore.hpp` | The app's flight logic, ROS-free: the module's policy, watchdog, handback order and parameter ranges |
| `structure_hold_app/src/structure_hold_app.cpp` | ROS 2 node: registers "Structure Hold" through `px4_ros2`, sends body rates + thrust at 100 Hz |
| `structure_hold_app/test/test_hold_core.cpp` | Unit tests for `HoldCore` (CI) |
| `package/` | `auterion-app.yml` and the AuterionOS Dockerfile (pinned to AuterionOS 4.6.2) |
| `stage_app.sh` | Writes a self-contained folder for `auterion-cli app build` |
| `docker/Dockerfile.sitl` | The app on stock ROS 2 Humble with the Micro XRCE-DDS Agent, for SITL |
| `sitl_app_test.sh`, `sitl_app_test.py` | Acceptance flight against stock PX4 v1.17.0 SIH |
| `tests/` | Package checks (pytest, CI) |

## Unit tests

```
cd sim2real/auterion/structure_hold_app/test
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal \
    -I../include -I../../../px4/structure_hold test_hold_core.cpp -o test_hold_core && ./test_hold_core
```

## Fly it in SITL (no Auterion hardware or account)

Needs Docker and a **stock** PX4 v1.17.0 tree built with `make px4_sitl_default` (do not
run `../px4/install.sh` into it: with the module compiled in, the module takes the first
external mode slot). PX4's SITL start script already runs the uXRCE-DDS client on UDP 8888.

```
pip install pymavlink
sim2real/auterion/sitl_app_test.sh ~/PX4-Autopilot          # builds the image, flies, writes runs/...
```

The app alone, against any PX4 on the same host: `docker run --rm --network host
structure-hold/app-sitl app -p vel_src:=1` (`vel_src` 1 only for SIH, whose velocity
estimate is unusable on v1.17.0). Settings are ROS 2 parameters named after the module's
`SH_` ones: `ros2 param set /structure_hold wd_est_to 5.0`; `test_fault 1` trips the
watchdog once.

## Build the AuterionOS app

```
sim2real/auterion/stage_app.sh /tmp/structure-hold-app
cd /tmp/structure-hold-app
pip install auterion-cli
auterion-cli app build                 # Skynode (arm64; needs arm64 Docker emulation)
auterion-cli app build --simulation    # Virtual Skynode (amd64)
auterion-cli app install build/*.auterionos
```

Not done yet: Virtual Skynode needs Auterion's Skynode Developer program (`docs/auterion-app.md`).
