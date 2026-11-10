import math

import pytest

from eval.baselines.run_baseline import (
    aggregate,
    compare,
    failsafe_in_window,
    log_line,
    row_from_summary,
)


def summary(scenario="wall1m_wind_8ms", seed=1, p95=0.2, toward=0.3, contacts=0,
            hover=None, ok=True):
    return {"scenario": scenario, "seed": seed, "run": f"runs/x_{seed}", "ok": ok,
            "hover": hover or {"passed": True},
            "metrics": {"p95_m": p95, "rms_m": p95 / 2, "max_m": 2 * p95,
                        "worst_toward_surface_m": toward, "min_clearance_m": 1 - toward,
                        "recovery_max_s": 3.0, "contacts": contacts,
                        "window_s": [25.0, 87.0]}}


def ekf(sequence, fraction=0.0):
    return {"modes": {"sequence": [{"t": t, "mode": m} for t, m in sequence],
                      "failsafe_fraction": fraction}}


CLEAN = [(0.0, "Hold"), (16.0, "Takeoff"), (23.0, "Hold"), (86.5, "Land"), (93.0, "Hold")]


def test_clean_flight_is_no_crash_no_failsafe():
    r = row_from_summary(summary(), ekf(CLEAN))
    assert not r["crash"] and not r["failsafe"] and not r["failsafe_in_hover"]
    assert r["p95_m"] == 0.2


def test_commanded_land_at_the_end_of_the_window_is_not_a_failsafe():
    assert failsafe_in_window(ekf(CLEAN)["modes"], [25.0, 87.0])  # raw check sees the Land
    assert not row_from_summary(summary(), ekf(CLEAN))["failsafe_in_hover"]


def test_failsafe_land_during_the_hover_counts():
    seq = [(0.0, "Hold"), (16.0, "Takeoff"), (23.0, "Hold"), (50.0, "Land"), (51.0, "Hold"),
           (86.5, "Land")]
    r = row_from_summary(summary(), ekf(seq, 0.02))
    assert r["failsafe"] and r["failsafe_in_hover"]


def test_contact_or_unfinished_flight_is_a_crash():
    assert row_from_summary(summary(contacts=1), ekf(CLEAN))["crash"]
    unfinished = summary(hover={"passed": False, "error": "TimeoutError: touchdown"})
    assert row_from_summary(unfinished, ekf(CLEAN))["crash"]
    no_log = {"scenario": "s", "seed": 1, "run": "r", "ok": False, "ulog_error": "no flight.ulg"}
    r = row_from_summary(no_log, None)
    assert r["crash"] and math.isnan(r["p95_m"])


def test_flown_but_unscored_flight_is_not_a_crash():
    unscored = {"scenario": "s", "seed": 1, "run": "r", "ok": False, "hover": {"passed": True},
                "ulog_error": "ImportError: reading .ulg needs pyulog"}
    r = row_from_summary(unscored, None)
    assert not r["crash"] and math.isnan(r["p95_m"])


def test_aggregate_and_compare():
    rows = [row_from_summary(summary(seed=s, p95=p, toward=t), ekf(CLEAN))
            for s, p, t in ((1, 0.2, 0.3), (2, 0.4, 0.5))]
    agg = aggregate(rows)["wall1m_wind_8ms"]
    assert agg["n"] == 2 and agg["crashes"] == 0
    assert math.isclose(agg["p95_m_mean"], 0.3) and agg["p95_m_worst"] == 0.4
    assert math.isclose(agg["min_clearance_m_worst"], 0.5)  # smallest clearance
    a = {"label": "stock", "aggregate": {"wall1m_wind_8ms": agg}}
    b = {"label": "tuned", "aggregate": {"wall1m_wind_8ms": {**agg, "p95_m_mean": 0.15}}}
    assert "-50 %" in compare(a, b)


def test_never_armed_is_not_flown_and_not_a_crash():
    s = {"scenario": "s", "seed": 1, "run": "r", "ok": False,
         "hover": {"passed": False, "error": "TimeoutError: vehicle never became armable"}}
    r = row_from_summary(s, None)
    assert not r["flown"] and not r["crash"]


def test_hold_runner_rows_read_the_flight_summary_and_expect_external_mode():
    # sim2real.px4.run_gazebo writes "flight" instead of "hover"; the scored window is
    # flown in External 1, which is not a failsafe for the hold runner.
    seq = [(0.0, "Hold"), (16.0, "Takeoff"), (23.0, "Position"), (28.0, "External 1"),
           (90.0, "Position"), (96.0, "External 1"), (98.0, "Position"), (100.0, "Land")]
    s = summary(hover={"passed": True})
    s["flight"] = s.pop("hover")
    s["metrics"]["window_s"] = [30.0, 89.9]
    r = row_from_summary(s, ekf(seq), "External 1")
    assert not r["crash"] and not r["failsafe_in_hover"]
    # The same window judged against Hold (the stock runner) would count it as a failsafe.
    assert row_from_summary(s, ekf(seq), "Hold")["failsafe_in_hover"]
    # A failsafe Land inside the hold still counts.
    seq_fs = seq[:4] + [(50.0, "Land"), (52.0, "External 1")] + seq[4:]
    assert row_from_summary(s, ekf(seq_fs, 0.02), "External 1")["failsafe_in_hover"]


def test_pool_merges_seed_chunks_and_prints_the_spread(tmp_path):
    import pytest

    from eval.baselines.run_baseline import pool, spread

    def batch(seeds, label="v10", p95=0.3):
        rows = [row_from_summary(summary(seed=s, p95=p95 * s), ekf(CLEAN)) for s in seeds]
        return {"label": label, "params": "p", "commit": "abc", "seeds": list(seeds),
                "rows": rows, "aggregate": aggregate(rows)}

    pooled = pool([batch([1, 2, 3]), batch([4, 5])], tmp_path / "pooled.json")
    assert pooled["seeds"] == [1, 2, 3, 4, 5]
    assert pooled["aggregate"]["wall1m_wind_8ms"]["n"] == 5
    assert abs(pooled["aggregate"]["wall1m_wind_8ms"]["p95_m_mean"] - 0.9) < 1e-9
    assert (tmp_path / "pooled.json").exists()
    text = spread(pooled)
    assert "n = 5 seeds" in text and "| seed 4 |" in text and "1.500" in text

    with pytest.raises(ValueError):
        pool([batch([1]), batch([1])], tmp_path / "dup.json")
    with pytest.raises(ValueError):
        pool([batch([1]), batch([2], label="other")], tmp_path / "mixed.json")


def test_log_line_names_the_profile_the_runner_flew():
    r = row_from_summary(summary(), ekf(CLEAN))
    batch = {"label": "tuned_v1", "commit": "abc1234", "params": "p.params", "run_args": []}
    assert ", vio_default estimate," in log_line(r, batch, "2026-10-23")
    for args in (["--profile", "vio_after_takeoff"], ["--profile=vio_after_takeoff"]):
        line = log_line(r, {**batch, "run_args": args}, "2026-10-23")
        assert ", vio_after_takeoff estimate," in line and "vio_default" not in line


def test_hold_runner_scores_the_hold_and_the_post_hold_sequence_apart():
    # The hold ran its 60 s; the sequence after it failed (stock PX4 struck the wall).
    s = summary(hover={"passed": False, "checks": {}, "error": "TimeoutError: re-engaged"})
    s["flight"] = s.pop("hover")
    s["metrics"]["window_s"] = [35.0, 93.0]
    r = row_from_summary(s, ekf([(0.0, "Hold")]), "External 1")
    assert not r["crash"] and r["post_hold_fail"]
    assert aggregate([r])["wall1m_wind_8ms"]["post_hold_fails"] == 1
    assert "post-hold sequence failed" in log_line(r, {"label": "x", "commit": "c",
                                                        "params": "p"}, "now")
    # The hold itself ended early: a crash, as on the stock runner.
    s["metrics"]["window_s"] = [35.0, 70.0]
    r = row_from_summary(s, ekf([(0.0, "Hold")]), "External 1")
    assert r["crash"] and not r["post_hold_fail"]
    # A contact inside the hold is a crash whatever follows.
    s["metrics"]["window_s"], s["metrics"]["contacts"] = [35.0, 93.0], 2
    assert row_from_summary(s, ekf([(0.0, "Hold")]), "External 1")["crash"]
    # The stock runner keeps its rule: an unfinished flight is a crash.
    u = summary(hover={"passed": False, "error": "TimeoutError: touchdown"})
    assert row_from_summary(u, ekf(CLEAN))["crash"]


def test_hold_runner_reads_the_window_where_hold_summary_keeps_it():
    # run_gazebo writes the window under "hold", not in "metrics"; a full hold whose
    # summary has it only there is not a crash (every v12 flight of 2026-10-23 was scored
    # a crash before this).
    s = summary()
    s["flight"] = s.pop("hover")
    s["hold"] = {"window_s": s["metrics"].pop("window_s")}
    s["hold"]["window_s"] = [30.2, 89.0]
    r = row_from_summary(s, ekf([(0.0, "Hold")]), "External 1")
    assert r["window_s"] == [30.2, 89.0]
    assert not r["crash"] and not r["post_hold_fail"]


def test_cpu_slots_hand_each_parallel_flight_its_own_cpus():
    from eval.baselines.run_baseline import cpu_slots

    cpus = list(range(10))
    assert cpu_slots(1, 4, cpus) == [None]  # one at a time: no pinning, as before
    assert cpu_slots(2, 4, cpus) == ["0,1,2,3", "4,5,6,7"]
    assert cpu_slots("auto", 3, cpus) == ["0,1,2", "3,4,5", "6,7,8"]
    assert cpu_slots("auto", 16, cpus) == [None]  # too few CPUs for one block: one flight
    with pytest.raises(ValueError):
        cpu_slots(3, 4, cpus)  # would oversubscribe
    with pytest.raises(ValueError):
        cpu_slots(0, 4, cpus)


def test_parallel_batch_keeps_flight_order_and_pins_each_flight(tmp_path, monkeypatch):
    import json
    import random
    import threading
    import time
    from argparse import Namespace
    from pathlib import Path

    import eval.baselines.run_baseline as rb

    monkeypatch.setattr(rb, "REPO_ROOT", tmp_path)
    (tmp_path / "p.params").write_text("")
    calls, active, peak = [], [0], [0]
    lock = threading.Lock()

    def fake_fly(scenario, seed, params, extra, runner, cpuset, container):
        with lock:
            calls.append((scenario, seed, cpuset, container))
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(random.uniform(0.0, 0.05))  # finish out of order
        folder = tmp_path / f"run_{scenario}_{seed}"
        folder.mkdir()
        (folder / "estimate_summary.json").write_text(json.dumps(summary(scenario, seed)))
        (folder / "ekf.json").write_text(json.dumps(ekf(CLEAN)))
        with lock:
            active[0] -= 1
        return Path(folder)

    monkeypatch.setattr(rb, "fly", fake_fly)
    monkeypatch.setattr(rb, "cpu_slots", lambda jobs, k: ["0,1", "2,3"])
    args = Namespace(params=tmp_path / "p.params", label="t", runner="estimate", run_args=[],
                     scenarios=["a", "b"], seeds=[1, 2, 3], jobs=2, cpus_per_job=2)
    batch = rb.run_batch(args)
    assert [(r["scenario"], r["seed"]) for r in batch["rows"]] == \
        [("a", 1), ("a", 2), ("a", 3), ("b", 1), ("b", 2), ("b", 3)]
    assert peak[0] <= 2 and batch["jobs"] == 2 and "2 flights in parallel" in batch["host"]
    assert {c[2] for c in calls} <= {"0,1", "2,3"}
    assert all(c[3] and c[3].startswith("px4-sitl-") for c in calls)
    out = next(tmp_path.glob("runs/*_px4_baseline_t_*"))
    lines = (out / "log_lines.md").read_text().splitlines()
    assert len(lines) == 6 and "seed 1" in lines[0] and " b seed 3" in lines[-1].replace(":", "")


# Surface passes (docs/surface-passes.md "In Gazebo"): the stock runner flies the pass in
# Offboard for the scenario's time and says so in its summary.
PASS_SEQ = [(0.0, "Hold"), (16.0, "Takeoff"), (23.0, "Hold"), (60.1, "Offboard"),
            (90.1, "Land"), (97.0, "Offboard")]


def pass_summary(window=(62.1, 90.1), **kw):
    s = summary(scenario="wall1m_pass_wind_8ms", **kw)
    s.update(scored_mode="Offboard", hold_s=30.0)
    s["metrics"]["window_s"] = list(window)
    s["metrics"]["pass"] = {"travel_m": 5.0, "standoff_p95_m": 0.2, "along_p95_m": 0.3,
                            "p95_while_moving_m": 0.35, "p95_while_still_m": 0.25,
                            "end_error_m": 0.1, "moving_s": 11.0}
    return s


def test_stock_pass_row_expects_offboard_and_the_scenarios_hold_time():
    r = row_from_summary(pass_summary(), ekf(PASS_SEQ))  # the stock runner's default: Hold
    assert not r["crash"] and not r["failsafe_in_hover"] and not r["post_hold_fail"]
    assert r["pass"]["standoff_p95_m"] == 0.2 and "moving_s" not in r["pass"]
    # Offboard lost half way (a failsafe to Hold) is a failsafe in the window and, with the
    # window cut short, a crash: the pass was not flown.
    lost = PASS_SEQ[:4] + [(75.0, "Hold"), (90.1, "Land")]
    cut = row_from_summary(pass_summary(window=(62.1, 75.0)), ekf(lost, 0.1))
    assert cut["crash"] and cut["failsafe_in_hover"]
    # A contact in a pass flown to the end is a crash too.
    assert row_from_summary(pass_summary(contacts=1), ekf(PASS_SEQ))["crash"]


def test_module_pass_row_reads_the_pass_numbers_from_the_hold():
    s = summary(scenario="wall1m_pass_wind_8ms")
    s["flight"] = s.pop("hover")
    s["hold_s"] = 30.0
    window = s["metrics"].pop("window_s")
    s["hold"] = {"window_s": [62.0, 90.0], "pass": {"travel_m": 5.0, "standoff_p95_m": 0.1,
                                                    "along_p95_m": 0.2}}
    assert window  # the stock summary keeps it in the metrics; the hold runner under "hold"
    seq = [(0.0, "Hold"), (16.0, "Takeoff"), (23.0, "Position"), (60.0, "External 1"),
           (90.0, "Position")]
    # 30 s of hold is the whole hold here, not half of run_gazebo's usual 60 s.
    r = row_from_summary(s, ekf(seq), "External 1", 60.0)
    assert not r["crash"] and r["pass"]["along_p95_m"] == 0.2
    assert r["pass"]["p95_while_moving_m"] is None


def test_table_lists_the_pass_numbers_when_a_batch_flew_passes():
    from eval.baselines.run_baseline import table

    rows = [row_from_summary(pass_summary(seed=s), ekf(PASS_SEQ)) for s in (1, 2)]
    batch = {"label": "tuned", "params": "p.params", "commit": "abc", "rows": rows,
             "aggregate": aggregate(rows)}
    text = table(batch)
    assert "standoff p95 m" in text and "| wall1m_pass_wind_8ms | mean of 2 |" in text
    plain = [row_from_summary(summary(seed=1), ekf(CLEAN))]
    assert "standoff" not in table({**batch, "rows": plain, "aggregate": aggregate(plain)})


def test_rows_carry_what_the_reference_took_out():
    s = summary(p95=0.2)
    s["metrics"].update(p95_logged_setpoint_m=0.5, ground_offset_m=0.3, setpoint_moved_m=1.2)
    row = row_from_summary(s, ekf(CLEAN))
    assert (row["p95_logged_setpoint_m"], row["ground_offset_m"]) == (0.5, 0.3)
    agg = aggregate([row])["wall1m_wind_8ms"]
    assert agg["p95_logged_setpoint_m_mean"] == 0.5 and agg["setpoint_moved_m_worst"] == 1.2
    # A batch scored before 2026-11-10, or a log with no stretch on the ground: NaN.
    old = row_from_summary(summary(), ekf(CLEAN))
    assert math.isnan(old["p95_logged_setpoint_m"])
    s["metrics"]["ground_offset_m"] = None
    assert math.isnan(row_from_summary(s, ekf(CLEAN))["ground_offset_m"])


def test_rescore_rewrites_the_batch_from_each_runners_rescore(tmp_path, monkeypatch):
    import json
    import sys
    import types

    from eval.baselines import run_baseline

    monkeypatch.setattr(run_baseline, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_baseline, "git_commit", lambda: "new1234")
    rows = [row_from_summary(summary(seed=s, p95=0.5), ekf(CLEAN)) for s in (1, 2)]
    for r in rows:
        (tmp_path / r["run"]).mkdir(parents=True)
        (tmp_path / r["run"] / "ekf.json").write_text(json.dumps(ekf(CLEAN)))
    (tmp_path / "runs" / "b").mkdir()
    batch_file = tmp_path / "runs" / "b" / "batch.json"
    batch_file.write_text(json.dumps({"label": "tuned", "params": "p", "commit": "old",
                                      "rows": rows, "aggregate": aggregate(rows)}))

    def again(out):
        if out.name == "x_2":
            raise FileNotFoundError("flight.ulg")  # the log is gone: the row stays
        s = summary(seed=1, p95=0.3)
        s["metrics"].update(p95_logged_setpoint_m=0.5, ground_offset_m=0.2,
                            setpoint_moved_m=0.0)
        return s

    fake = types.ModuleType("fake_runner")
    fake.rescore = again
    monkeypatch.setitem(sys.modules, "fake_runner", fake)
    monkeypatch.setitem(run_baseline.RUNNERS, "estimate",
                        ("fake_runner", "estimate_summary.json", "Hold"))
    batch = run_baseline.rescore(batch_file)
    assert [r["p95_m"] for r in batch["rows"]] == [0.3, 0.5]
    assert batch["rows"][0]["p95_logged_setpoint_m"] == 0.5
    assert batch["rescored_at"] == "new1234"
    assert json.loads(batch_file.read_text())["aggregate"]["wall1m_wind_8ms"][
        "p95_m_mean"] == pytest.approx(0.4)
    assert "0.300" in (batch_file.parent / "table.md").read_text()
