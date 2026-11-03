import json

import pytest
import yaml

from eval.field_faults import CONTACT_M, score_run, table
from sim2real.estimate_bridge.bridge import TraceWriter
from sim2real.estimate_bridge.config import load_profile


def fake_run(tmp_path, runner="hold", toward_wall_mps=0.0, trip=True):
    """A run folder as run_gazebo / run_estimate leave it: 1 m from the wall1m wall (x = 0,
    normal +x), engaged at 10 s, the field fault 20 s later; after it the truth walks toward
    the wall at ``toward_wall_mps``."""
    prof = load_profile("field_drift_at_engage")
    run = tmp_path / f"20261101_sitl_{'hold' if runner == 'hold' else 'estimate'}_x"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(
        {"scenario": "wall1m_wind_4ms", "profile": prof.to_dict()}))
    (run / "estimate_verify.json").write_text(json.dumps({"anchor_t": 10.0}))
    if runner == "hold":
        lines = ["WARN  [structure_hold] Structure Hold: engage refused: landed",
                 "WARN  [structure_hold] watchdog refusal: landed, handing back"]
        if trip:
            lines.append("WARN  [structure_hold] Structure Hold: watchdog trip: position_error,"
                         " handing back")
        (run / "px4.log").write_text("\n".join(lines) + "\n")
    bridge = prof.bridge()
    w = TraceWriter(run / "estimate_trace.csv")
    for k in range(int(70.0 / 0.02)):
        t = 0.02 * k
        z = min(t, 2.5)
        x = -1.0 + toward_wall_mps * max(0.0, t - 30.0)
        w.write(bridge.update(t, [x, 0.0, z], engaged=t >= 10.0))
    w.close()
    return run


def test_a_vehicle_that_walks_into_the_wall_after_the_fault_is_a_contact(tmp_path):
    r = score_run(fake_run(tmp_path, toward_wall_mps=0.04))
    assert r["t_fault_s"] == pytest.approx(30.0)
    assert r["clearance_before_m"] == pytest.approx(1.0, abs=1e-6)
    assert r["min_clearance_m"] == pytest.approx(0.0, abs=0.01)  # 25 s x 4 cm/s
    assert r["contact"] and r["max_horizontal_m"] == pytest.approx(1.0, abs=0.01)
    assert r["trip"] == "position_error" and r["airborne_end"]


def test_a_steady_vehicle_is_no_contact_and_stock_has_no_trip(tmp_path):
    r = score_run(fake_run(tmp_path, runner="stock"))
    assert not r["contact"] and r["min_clearance_m"] > CONTACT_M
    assert r["max_horizontal_m"] == pytest.approx(0.0, abs=1e-9)
    assert r["runner"] == "stock" and r["trip"] is None
    assert "field_drift_at_engage" in table([r])


def test_refusals_and_the_sequence_test_fault_are_not_trips(tmp_path):
    run = fake_run(tmp_path, trip=False)
    assert score_run(run)["trip"] == "none"
    with open(run / "px4.log", "a") as f:
        f.write("WARN  [structure_hold] Structure Hold: watchdog trip: test_fault, handing back\n")
    assert score_run(run)["trip"] == "none"


def test_a_profile_without_field_faults_is_refused(tmp_path):
    run = fake_run(tmp_path)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    cfg["profile"]["field_faults"] = []
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        score_run(run)


def test_an_anchor_that_never_fired_is_reported(tmp_path):
    run = fake_run(tmp_path)
    (run / "estimate_verify.json").write_text(json.dumps({"anchor_t": None}))
    assert "error" in score_run(run)


def test_a_trace_gap_over_the_whole_window_is_reported_not_scored(tmp_path):
    run = fake_run(tmp_path)
    path = run / "estimate_trace.csv"
    head, *rows = path.read_text().splitlines()
    kept = [r for r in rows if not (30.0 <= float(r.split(",")[1]) < 62.0)]
    path.write_text("\n".join([head, *kept]) + "\n")
    r = score_run(run)
    assert "error" in r and "no trace samples" in r["error"]
