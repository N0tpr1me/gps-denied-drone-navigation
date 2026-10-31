# Why v19 looked 0.2 m off its hold point

`docs/vio-fit.md` found that with the estimate error fitted to real VIO, v19's Gate 1 p95 sits
at about 0.21 m in every cell, calm or windy, and the cascaded PD (the mock's yardstick, not
tuned PX4) holds up to 22 % tighter in calm air and at 4 m/s. This page is what sets that
number and what was tried. Short answer: **it is not a standing offset; it is how slowly v19
settles after engage**, and that slowness is what makes it robust to estimate jumps.

## What the 0.2 m is made of

Gate 1 scores the p95 of the whole 20 s episode, and every episode starts with the drone
0.2 m per axis (about 0.3 m in all) off the hold point (`start_offset_std_m`). 5 % of 20 s is
1 s, so the p95 is set by the first second or two. `eval/settle_split.py` splits the error into
the settle and the steady hold (after 5 s). Gate 1 x500 cells through the fusion, fitted VIO
estimate (`eval/configs/gate1_x500_ekf_vio_fit_v19.yaml`), scratch seed 8001, 256 episodes:

| Cell | v19 p95, whole episode | PD | v19 p95 after 5 s | PD after 5 s | Estimate's own error, p95 after 5 s |
| --- | --- | --- | --- | --- | --- |
| 0 m/s, 1 m | 0.194 | 0.161 | 0.103 | 0.099 | 0.080 |
| 4 m/s, 1 m | 0.194 | 0.173 | 0.104 | 0.115 | 0.080 |
| 8 m/s, 2 m | 0.212 | 0.307 | 0.120 | 0.222 | 0.080 |

Once settled, v19 holds as tight as the PD in calm air and tighter in wind; both sit just
above the estimate's own error. Its mean steady offset is 0.04 m back from the wall (the
`toward_wall` penalty) and under 0.01 m on the other axes, so there is no integrator problem:
the mean wind moves neither.

The difference is the settle. Mean error after engage, calm, 1 m, estimate perfect (no
corruption at all, so only the controller is left):

| Time after engage | 0.5 s | 1 s | 1.5 s | 2 s | 3 s | 4 s | 6 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v19 | 0.255 | 0.178 | 0.147 | 0.126 | 0.083 | 0.066 | 0.051 |
| PD | 0.269 | 0.165 | 0.076 | 0.019 | 0.055 | 0.075 | 0.052 |
| v10 | 0.276 | 0.220 | 0.175 | 0.139 | 0.093 | 0.065 | 0.038 |
| v12 | 0.264 | 0.205 | 0.170 | 0.152 | 0.118 | 0.098 | 0.071 |
| v17 | 0.237 | 0.155 | 0.131 | 0.110 | 0.080 | 0.063 | 0.047 |
| v18 | 0.248 | 0.180 | 0.147 | 0.119 | 0.083 | 0.062 | 0.045 |

The PD is in by 2 s (and then wanders with the gusts). v19 closes the last 0.15 m with a time
constant of about 2 s, and so has every PPO policy since v10. It is the same with v19's
reward change (action_band) and without it, so it is not that term.

## Why it settles slowly

To the policy an offset in its estimate is either real (a start or re-engage) or an estimate
jump that will decay back over about 5 s, and after the 180 ms history has scrolled past a
step it cannot tell which. In training jumps came at 0.1 Hz and re-engages at 0.05 Hz, so an
offset was twice as likely to be fake, and chasing a fake one moves the true position away by
the jump. The policy learned to close any offset slowly.

The test (v21, below) bears this out: a policy taught that real offsets are more common
settles faster, and in exchange follows jumps more and holds looser on the standard estimate.

## v21: more real offsets in training (not a candidate)

`training/configs/structure_hold_ppo_v21.yaml`: v19's recipe with re-engages at 0.2 Hz (a real
offset twice as often as a jump), jumps and dropouts unchanged. v19's `policy.pt` went with
its run folder, so the run starts the actor from the released ONNX (`init_actor_onnx`) and
trains a fresh critic alone for 200 iterations first. Run
`20261030-205514_structure_hold_ppo_v21_s0_4241999`, 800 iterations after the warm-up, 18 min.
Its checkpoint selection (standard eval corruption) picked iteration 200, which is v19 itself:
every fine-tuned checkpoint scored worse (p95 0.29 to 0.42 m against 0.25, crash 0.8 to 2.0 %
against 0.4 %, 256 episodes). With `eval.settle_split` (scratch seed 8001):

| | v19 | v21 it 450 | v21 it 999 | PD |
| --- | --- | --- | --- | --- |
| Fitted VIO, calm 1 m: p95 whole episode | 0.194 | 0.171 | 0.185 | 0.161 |
| Fitted VIO, calm 1 m: error at 2 s | 0.132 | 0.088 | 0.098 | 0.030 |
| Fitted VIO, 8 m/s 2 m: p95 whole episode | 0.212 | 0.204 | 0.216 | 0.307 |
| Standard estimate, calm 1 m: p95 after 5 s | 0.148 | 0.171 | 0.170 | 0.247 |
| Standard estimate, 8 m/s 1 m: p95 after 5 s | 0.156 | 0.180 | 0.181 | 0.295 |

Faster settle (2 s error down by a third), but 15 % looser steady hold on the standard
estimate, which is what Gate 1, Gate 2 and the Gazebo-like check fly. Not put through
`eval/configs/gazebo_check.yaml`: its rough cells are the standard estimate, where it is
already worse. v19 stays.

## v22: the time-since-engage input (released 2026-10-31)

The fix proposed here, tried. The observation gets a 21st value, seconds since the last
engage (episode start or re-engage) over `engage_obs_cap_s` (4 s), capped at 1
(`envs/structure_hold/env.py`; policies up to v21 still read the first 20). v22
(`training/configs/structure_hold_ppo_v22.yaml`, run `20261030-220041`) is v19's ONNX with
zero weights on the new input, so it starts out flying exactly as v19, plus a 200-iteration
critic warm-up and 1000 PPO iterations on v19's recipe (re-engages back at 0.05 Hz), 23 min.

Scratch seed 8001 (`eval.settle_split`, 256 episodes, metres):

| | v19 | v22 | PD |
| --- | --- | --- | --- |
| Fitted VIO, calm 1 m: p95 / after 5 s / error at 2 s | 0.194 / 0.103 / 0.132 | 0.157 / 0.090 / 0.084 | 0.161 / 0.099 / 0.030 |
| Fitted VIO, 8 m/s 1 m: p95 / after 5 s | 0.200 / 0.108 | 0.172 / 0.106 | 0.227 / 0.155 |
| Standard estimate, calm 1 m: p95 / after 5 s | 0.217 / 0.141 | 0.169 / 0.110 | 0.303 / 0.260 |
| Standard estimate, 8 m/s 1 m: p95 / after 5 s | 0.224 / 0.147 | 0.184 / 0.126 | 0.334 / 0.280 |

It settles faster and, unlike v21, holds tighter after the settle as well, so it does not
buy the settle by following jumps. Against the release gates (`experiments/LOG.md`):

| Check | v19 | v22 |
| --- | --- | --- |
| Gazebo-like check (`gazebo_check_v22.yaml`, run `20261030-224041`), rough cells pooled p95 | 0.244 m | 0.196 m: beats v19, ratio 0.81 (0.79 to 0.83), no rule failed; calm clean 0.037 vs 0.044 m, seed 3 replay toward the wall 0.19 vs 0.24 m, fast ring 0.017 rad/s, 0.02 m low |
| Gate 1 x500 fused, held-out (run `20261030-230516`) | 6/6, pooled 0.241 m, headline -34 %, 7 crashes in 18,432 | 6/6, pooled 0.19 m vs PD 0.36, headline -47 %, no crash |
| Gate 1 stress (run `20261030-232614`) | 13/13, pooled 0.256 m | 13/13, pooled 0.21 m vs PD 0.38, crash 0.1 % vs 0.6 %; the 5 s dropout cell (outside the envelope, decided 2026-10-24) still loses |
| Fitted-VIO re-check (run `20261031-000742`) | 1/6, pooled 0.211 vs PD 0.213 m | 5/6, pooled 0.18 vs 0.21 m; the headline cell is -23 % against the -30 % rule |

A check ratio of 0.81 is about where v11, v12 and v18 stood before Gazebo flew them worse,
so the Gazebo re-fly decides. The PX4 module reads the input when its header has it
(`HAS_ENGAGE_TIME`; `sample.engage_s` from the engage time in `StructureHold.cpp`), and
`tests/test_policy.cpp` passes on the v22 header (observation within 3e-7 of the env, MLP
within 1e-6 of onnxruntime).

Gazebo confirmed it (`docs/baseline.md`, "v22 candidate vs tuned PX4 and v19"): over 8 seeds
mean hold p95 0.198 / 0.196 / 0.194 m at calm / 4 / 8 m/s against v19's 0.221 / 0.224 /
0.227 and tuned PX4's 0.427 / 0.424 / 0.428 (-54 / -54 / -55 %), lower than v19 in all 24
seed and wind cells, no crash or contact. Zain released it on 2026-10-31 and the module
header is generated from it. To watch on a real flight: after the scripted re-engage in the
seed 3 near-wall flow run it came to 0.54 m from the wall (v19 0.75 m) while the wall guard
moved the hold point, and its pitch-rate content is a little higher than v19's.

## Why the input (the reasoning before v22)

Reward weights cannot: the trade is between two things the policy cannot tell apart. Giving it
the missing fact can. The module knows when it engaged, so a **time-since-engage input**
(capped at a few seconds) would let the policy trust the offset it starts with and keep
distrusting steps later. Cost: the observation goes from 20 to 21 inputs, so the module's
contract and `PolicyWeights.hpp` change, and a new policy has to pass the Gazebo-like check,
Gate 1 and a Gazebo re-fly before replacing v19 (a Decided change, Zain's call). The new
first-layer column can start at zero, so training continues from v19's actor
(`init_actor_onnx` with a padded first layer).

How much it is worth: the start transient weighs this much because Gate 1 episodes are 20 s
long. In the 60 s Gazebo hold (Gate 2) the same settle is a much smaller share of the window.
It matters most for the Gate 1 re-check on the fitted VIO model, and for any customer test
that scores the first seconds after engage.
