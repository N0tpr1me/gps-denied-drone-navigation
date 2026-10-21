# Released mock-simulator policies

| File | Run | Result (256 envs x 20 s, default wind and estimate corruption) |
| --- | --- | --- |
| `structure_hold_v0.onnx` + `.json` | `20261021-204332_structure_hold_ppo_s0_8567e0e` (1500 PPO iterations, 20 min on 4 CPU cores) | p95 0.56 m vs cascaded PD 0.71 m; crash rate 31 % vs 6.6 %. Crashes are tilt-overs after estimate jumps or dropouts (1 % with a clean estimate), so this policy is a training-loop proof, not a flyable one. |

| `structure_hold_v1.onnx` + `.json` | `20261021-220953_structure_hold_ppo_s0_3a46cff` (bounded reward with tilt penalty, trained on 5x the standard dropout and jump rates, checkpoint 750 of 1500 selected for the lowest crash rate on a selection seed) | p95 0.52 m vs PD 0.71 m; crash rate 8.9 % vs 6.6 % on the reporting seed (4.7 % on the selection seed; 256 episodes give about +/- 3 points). Crashes are now drifts out of the 3 m box during long dropouts, no tilt-overs. |

Decisive comparison of v1 (run `20261021-232411_structure_hold_eval_s1_9f99246`, 3 seeds x
2048 identical 20 s episodes for the policy and the PD, standard wind and corruption): p95
0.48 m vs PD 0.83 m; crash rate 6.1 % +/- 0.3 vs 3.8 % +/- 0.2 (binomial s.e.). The policy
holds much tighter and crashes more, about eight standard errors apart, so the gap is real.

Score any of them: `python -m training.launch --config training/configs/structure_hold_eval.yaml --device cpu`.
