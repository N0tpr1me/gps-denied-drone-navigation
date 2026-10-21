# Released mock-simulator policies

| File | Run | Result (256 envs x 20 s, default wind and estimate corruption) |
| --- | --- | --- |
| `structure_hold_v0.onnx` + `.json` | `20261021-204332_structure_hold_ppo_s0_8567e0e` (1500 PPO iterations, 20 min on 4 CPU cores) | p95 0.56 m vs cascaded PD 0.71 m; crash rate 31 % vs 6.6 %. Crashes are tilt-overs after estimate jumps or dropouts (1 % with a clean estimate), so this policy is a training-loop proof, not a flyable one. |

Score any of them: `python -m training.launch --config training/configs/structure_hold_eval.yaml --device cpu`.
