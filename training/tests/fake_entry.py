"""Entrypoint used by the launch tests: behaves as params.mode says."""

import sys
import time

from training.runlib import RunWriter, entry_args, load_yaml


def main() -> int:
    args = entry_args()
    params = load_yaml(args.config)["params"]
    mode = params.get("mode", "ok")
    print(f"fake_entry mode={mode} smoke={args.smoke} device={args.device}")
    writer = RunWriter(args.run_dir)
    writer.metric(0, value=1.0)
    if mode == "ok":
        writer.summary("ok", headline={"value": 1.0, "smoke_seen": args.smoke},
                       vs_baseline="+0% vs fake")
        return 0
    if mode == "fail":
        writer.summary("failed", error="fake failure")
        return 1
    if mode == "crash":
        raise RuntimeError("fake crash without summary")
    if mode == "no_summary":
        return 0
    if mode == "sleep":
        time.sleep(60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
