"""Environment self-check: proves a machine can run a launched job end to end.

Fits a small linear model by gradient descent and compares it with the closed-form
least-squares answer (the "classical baseline" for this toy problem). It exercises the
whole launch path (config, run folder, metrics, summary, LOG.md) with no simulator, and
reports which optional pieces (PyTorch, CUDA) the machine has. The first real training
entrypoint replaces it as the thing people launch; it stays as a machine check.
"""

from __future__ import annotations

import platform
import sys
import traceback

import numpy as np

from training.runlib import RunWriter, entry_args, load_yaml


def torch_info(device: str) -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"torch": "not installed"}
    info: dict[str, object] = {"torch": torch.__version__}
    if device == "cuda":
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def run(params: dict, seed: int, writer: RunWriter, log_every: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n, d = int(params["samples"]), int(params["features"])
    x = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = x @ w_true + float(params["noise"]) * rng.normal(size=n)

    w_lstsq, *_ = np.linalg.lstsq(x, y, rcond=None)
    baseline_loss = float(np.mean((x @ w_lstsq - y) ** 2))

    w = np.zeros(d)
    lr = float(params["learning_rate"])
    loss = float("nan")
    for it in range(int(params["iterations"])):
        err = x @ w - y
        loss = float(np.mean(err**2))
        if not np.isfinite(loss):
            raise FloatingPointError(f"loss became {loss} at iteration {it}")
        w -= lr * (2.0 / n) * (x.T @ err)
        if it % log_every == 0:
            writer.metric(it, loss=loss)
    writer.metric(int(params["iterations"]), loss=loss)
    return {"final_loss": loss, "baseline_loss": baseline_loss}


def main(argv: list[str] | None = None) -> int:
    args = entry_args(argv)
    cfg = load_yaml(args.config)
    writer = RunWriter(args.run_dir)
    try:
        print(f"selfcheck: python {sys.version.split()[0]} on {platform.platform()}")
        print(f"selfcheck: numpy {np.__version__}, {torch_info(args.device)}")
        params = cfg["params"]
        result = run(params, int(cfg["seed"]), writer, log_every=max(1, params["iterations"] // 20))
        gap = result["final_loss"] / result["baseline_loss"] - 1.0
        print(f"selfcheck: gradient descent loss {result['final_loss']:.6f}, "
              f"closed-form {result['baseline_loss']:.6f} ({gap:+.2%})")
        writer.summary(
            "ok",
            headline={**result, "iterations": params["iterations"]},
            vs_baseline=f"{gap:+.2%} loss vs closed-form least squares",
        )
        return 0
    except Exception as e:  # report every failure in summary.json, then exit non-zero
        traceback.print_exc()
        writer.summary("failed", error=f"{type(e).__name__}: {e}")
        return 1
    finally:
        writer.close()


if __name__ == "__main__":
    sys.exit(main())
