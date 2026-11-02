"""The evaluation kit's plan, verdict rule and table, on fake run folders (no Docker)."""

import json

import pytest
import yaml

from eval import eval_kit
from eval.eval_kit import (
    FAULTS,
    Flight,
    comparison_table,
    estimate_minutes,
    judge,
    pair_rows,
    plan,
    score_all,
)
from eval.tests.test_field_faults import fake_run


def row(**kw):
    base = {"runner": "hold", "contact": False, "min_clearance_m": 0.9, "max_horizontal_m": 0.2,
            "max_climb_m": 0.0, "max_drop_m": 0.0, "min_height_m": 2.4, "airborne_end": True,
            "trip": "none"}
    return {**base, **kw}


def test_plan_flies_each_fault_with_both_controllers_on_matching_profiles():
    flights = plan(list(FAULTS))
    assert len(flights) == 10
    for f in flights:
        anchor = "engage" if f.runner == "hold" else "hover"
        assert f.run_args[:2] == ["--profile", f"field_{f.fault}_at_{anchor}"]
        assert ("--rangefinder" in f.run_args) == (f.runner == "hold")
    assert "--image" in plan(["reset"], "my/image:tag")[0].run_args
    with pytest.raises(ValueError):
        plan(["meteor"])


def test_time_estimate_shrinks_with_parallel_slots():
    flights = plan(list(FAULTS))
    assert estimate_minutes(flights, 10) == pytest.approx(2.2)
    assert estimate_minutes(flights, 1) == pytest.approx(20.0)


def test_verdict_rule_contact_then_clearance_then_movement():
    assert judge(row(contact=True), row(runner="stock"))[0] == "tuned PX4"
    assert judge(row(), row(runner="stock", contact=True))[0] == "Structure Hold"
    assert judge(row(min_clearance_m=0.99), row(min_clearance_m=0.75))[0] == "Structure Hold"
    assert judge(row(max_horizontal_m=0.5), row(max_horizontal_m=0.1))[0] == "tuned PX4"
    assert judge(row(), row(min_clearance_m=0.93))[0] == "about the same"
    assert judge({"error": "x"}, row())[0] == "not scored"


def test_kit_scores_fake_flights_and_prints_one_table(tmp_path):
    hold = fake_run(tmp_path)
    stock = fake_run(tmp_path, runner="stock", toward_wall_mps=0.04)
    flights = [Flight("drift", "hold", ["--profile", "field_drift_at_engage"], str(hold)),
               Flight("drift", "estimate", ["--profile", "field_drift_at_hover"], str(stock)),
               Flight("reset", "hold", ["--profile", "field_reset_at_engage"], None),
               Flight("reset", "estimate", ["--profile", "field_reset_at_hover"], None)]
    pairs = pair_rows(flights, score_all(flights))
    assert [p["better"] for p in pairs] == ["Structure Hold", "not scored"]
    assert pairs[0]["why"] == "tuned PX4 touched the wall"
    text = comparison_table(pairs, {"scenario": "wall1m_wind_4ms", "seed": 1, "commit": "abc"})
    assert "Structure Hold better on 1 of 2 faults" in text
    assert "Wall contacts: Structure Hold 0, tuned PX4 1" in text
    assert "tripped (position jumped) and handed back" in text


def test_rescore_reads_a_kit_folder(tmp_path, capsys):
    kit = tmp_path / "kit"
    kit.mkdir()
    (kit / "config.yaml").write_text(yaml.safe_dump({"scenario": "wall1m_wind_4ms", "seed": 1,
                                                     "commit": "abc"}))
    (kit / "flights.json").write_text(json.dumps([
        {"fault": "freeze", "runner": "hold", "run_args": ["--profile", "p"], "run": None},
        {"fault": "freeze", "runner": "estimate", "run_args": ["--profile", "p"], "run": None}]))
    assert eval_kit.main(["--rescore", str(kit)]) == 0
    assert "estimate freezes for 5 s" in (kit / "comparison.md").read_text()


def test_dry_run_needs_no_docker(capsys):
    assert eval_kit.main(["--dry-run", "--jobs", "1", "--faults", "reset"]) == 0
    assert "2 flights" in capsys.readouterr().out
