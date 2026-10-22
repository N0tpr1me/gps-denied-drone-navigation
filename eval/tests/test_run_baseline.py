import math

from eval.baselines.run_baseline import aggregate, compare, failsafe_in_window, row_from_summary


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
