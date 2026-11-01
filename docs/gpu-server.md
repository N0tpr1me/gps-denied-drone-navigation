# The GPU server: getting it ready and moving the simulations there

From 2026-10-31 the simulations run on the GPU server, not on the Mac (Zain). This page is
what to do when the server arrives, in order. Most of it is one script.

The server does two different jobs:

- **Training and Isaac Lab** use the GPUs (one free GPU per run; `training/launch.py`
  checks `nvidia-smi` and refuses busy GPUs).
- **The Gazebo + PX4 benchmark** (Gate 2 style flights against tuned PX4) uses **CPU cores
  only**. Gazebo runs without a screen and the x500 has no camera, so it never touches a
  GPU. On the server many flights run at once, each on its own 4 CPU cores, instead of one
  at a time on the Mac.

## 1. What the server's admin does once (needs sudo)

Ubuntu 22.04 or 24.04 on x86_64. Then:

1. **NVIDIA driver 580.65.06 or newer** (Isaac Sim 5.1 needs it):
   `sudo ubuntu-drivers install` or `sudo apt install nvidia-driver-580`, reboot, and check
   `nvidia-smi` lists every GPU.
2. **Docker**, usable without sudo:
   `curl -fsSL https://get.docker.com | sh` then `sudo usermod -aG docker $USER`, log out
   and back in, check `docker run --rm hello-world`.
3. **NVIDIA Container Toolkit**, so containers can see the GPUs (NVIDIA's install guide;
   on Ubuntu):

   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
     | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
     | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
   ```

4. `git`, `curl` and `tmux` (`sudo apt install git curl tmux`), and at least 80 GB free disk
   (SITL image about 10 GB, Isaac Lab image about 25 GB, plus runs).

Nothing else needs sudo. The setup script installs Python packages into a folder inside the
repo (`.venv`), never system-wide.

## 2. Giving Claude access

**Recommended: Claude Code with Remote Control on the server**, the same way the Mac works
today. Claude then works on the server from this project, with no open ports and no keys
to hand over.

```bash
curl -fsSL https://claude.ai/install.sh | bash      # installs Claude Code for this user
git clone https://github.com/N0tpr1me/gps-denied-drone-navigation.git
cd gps-denied-drone-navigation
tmux new -s claude                                   # keeps it running after you log out
claude                                               # sign in once with your account, then /exit
claude remote-control                                # the server now shows up as your device
```

Then tell the project "the server is up". A thread can start a session on the server and
run everything below. Detach from tmux with Ctrl-b then d; it keeps running.

**Not recommended: SSH from the cloud.** The cloud sessions would need the server reachable
from the internet, its address allowed in the cloud environment's network settings, and a
private key stored as a secret. That is more to set up and more to keep safe, for no gain
over Remote Control.

**Fallback: you run the commands.** Run the script below yourself and paste
`report.md` in the project; Claude reads it and tells you the next command.

## 3. One command: set up and test everything

```bash
cd gps-denied-drone-navigation
bash training/server_setup.sh            # about 1 hour the first time
```

It stops early and says what is missing if the driver, Docker or disk space is not right.
Otherwise it runs, in order: the machine report (`training/server_report.sh`), Python
env, lint and tests, a training smoke run on CPU and on one free GPU, the SITL image build
(PX4 v1.17.0 with the released policy compiled in, v22 today), a hover test, several
hovers at once to check each still runs at real time, and the Isaac Lab image pull. It
writes `runs/server_setup_<time>/report.md`; paste that in the project. Re-running it is
safe. `--no-isaac` skips the 25 GB Isaac Lab pull.

If the Isaac Lab pull says "unauthorized", make a free NVIDIA NGC account and run
`docker login nvcr.io` (user `$oauthtoken`, password your NGC API key), then re-run.

## 4. Running the benchmark on the server

The same commands as on the Mac, plus `--jobs auto`:

```bash
source .venv/bin/activate
python -m eval.baselines.run_baseline --runner hold --label v22_server \
    --params eval/baselines/px4_tuned_v1.params --seeds 1 2 3 4 5 6 7 8 --jobs auto \
    -- --profile vio_at_engage                  # the policy
python -m eval.baselines.run_baseline --label tuned_v1_server \
    --params eval/baselines/px4_tuned_v1.params --seeds 1 2 3 4 5 6 7 8 --jobs auto \
    -- --profile vio_at_hover                   # tuned PX4
```

- `--jobs auto` runs as many flights at once as the CPUs allow at 4 cores each
  (`--cpus-per-job` changes that); `--jobs 6` sets the number yourself. Each flight gets
  its own CPU cores, its own container name and its own Gazebo "partition", so flights
  cannot hear each other. Results come out in the same order as one at a time.
- Leave CPU cores free for training when both run: for example `--jobs 8` on a 64-core
  machine leaves half the cores.
- Every stock-runner row records Gazebo's real-time factor (`rtf_min` in `batch.json`).
  If it drops well below the Mac's (0.97 mean, 0.85 min), use fewer jobs.
- Before the server's numbers replace the Mac's in `docs/baseline.md`, fly the tuned PX4
  and v22 batches once on the server and compare with the Mac's 8-seed table: the
  machine changed, so the numbers can move a little (`docs/baseline.md` keeps the Mac's
  as the record). `experiments/LOG.md` lines name the host automatically
  (`SITL_HOST_LABEL` overrides it).

## 5. Isaac Lab

The plan in `docs/isaac-lab.md` fits this server as it is: the RTX 4500 Ada has RT cores
and 24 GB (Isaac Sim needs RT cores and 16 GB or more; A100 and H100 would not work),
Ubuntu 22.04/24.04, driver 580.65.06+, Docker with the NVIDIA Container Toolkit, 32 GB+
RAM. After `server_setup.sh` has pulled the image, follow its "First hour on the new
server" (steps 3 to 9): parity check, then a smoke training run on one GPU. Isaac Lab and
the SITL flights run in separate containers and do not clash.

## Rules on the server (the same as everywhere)

- `nvidia-smi` before any GPU job; only free GPUs (`launch.py` enforces it).
- Ask Zain before any run over 1 hour or on more than 1 GPU (`launch.py` enforces it;
  `--approved` only with his yes). Benchmark batches of SITL flights use no GPU; a batch
  that would take over an hour is split or asked about like any other run.
- Real runs go in `experiments/LOG.md`; nothing secret or over 50 MB in git.
- One Claude session per checkout. For two sessions at once, give the second its own
  `git worktree`.

## What has been checked without the server (2026-10-31)

In the Claude cloud container (Linux x86_64, 4 cores, no GPU, Docker): the SITL image
built with PX4 v1.17.0 and the v22 policy and passed the hover test at real-time factor
1.00; Structure Hold flights ran one at a time and two at once. Gazebo's apt host is
blocked there, so that build used PX4's published Gazebo Harmonic image as its base
(`sim2real/docker/cloud_build.sh`); the server builds the normal image with `run.sh build`.
Not checked until the server: the GPU steps (CUDA torch, GPU in Docker, Isaac Lab) and how
many flights a Threadripper runs at once at real time.
