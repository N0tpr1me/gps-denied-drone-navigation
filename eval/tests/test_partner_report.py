"""The one-page partner benchmark report, from a synthetic scored session."""

import json
import math
import statistics

import pytest
import yaml

from eval.partner_report import headline, main, render, strip_chart, summarise
from eval.partner_session import score_session
from eval.tests.test_partner_session import SESSION, START_US, flight


def scored(tmp_path, early=False):
    runs = [{"file": f"{case}_{c}_{i}.ulg", "controller": c, "case": case, "run": i}
            for case in ("calm", "wind_4ms") for i in (1, 2)
            for c in ("tuned_px4", "structure_hold")]
    runs.append({"file": "x.ulg", "controller": "tuned_px4", "case": "calm", "run": 3,
                 "aborted": True, "reason": "VIO lost tracking on the climb"})
    sheet = tmp_path / "session.yaml"
    sheet.write_text(yaml.safe_dump({"drone": "Test X4 <b>", "site": "lab", "date": "2027-05-01",
                                     **SESSION, "runs": runs}))
    wobble = {"tuned_px4": 0.2, "structure_hold": 0.1}

    def loader(path):
        run = next(r for r in runs if r["file"] == path.name)
        end = 40.0 if early and run["file"] == "calm_structure_hold_2.ulg" else 82.0
        return flight(run["controller"], wobble=wobble[run["controller"]], mode_end=end), \
            START_US

    return json.loads(json.dumps(score_session(sheet, loader), default=str))


def test_headline_and_numbers_come_from_the_scored_session(tmp_path):
    sm = summarise(scored(tmp_path))
    assert round(sm["change"]) == -50
    assert [c["case"] for c in sm["cases"]] == ["calm", "wind_4ms"]
    assert sm["per"]["structure_hold"]["n"] == 4 and sm["per"]["tuned_px4"]["n"] == 4
    assert "50 % tighter" in headline(sm)
    assert any("VIO lost tracking" in e for e in sm["events"])


def test_an_early_end_is_listed_in_plain_words(tmp_path):
    sm = summarise(scored(tmp_path, early=True))
    assert sm["per"]["structure_hold"]["early"] == 1
    text = " ".join(sm["events"])
    assert "position estimate jumped" in text and "Position mode" in text


def test_the_page_is_self_contained_and_escaped(tmp_path):
    page = render(scored(tmp_path), "abc1234", "2027-05-02 10:00 UTC")
    assert page.startswith("<!doctype html>") and "<script" not in page
    assert "Test X4 &lt;b&gt;" in page and "<b>\"" not in page
    assert page.count("<circle") == 4 and page.count("<rect") == 4  # one mark per hold
    assert "abc1234" in page and "-50 %" in page


def test_cli_writes_next_to_the_summary(tmp_path):
    summary = tmp_path / "session_summary.json"
    summary.write_text(json.dumps(scored(tmp_path)))
    assert main([str(summary)]) == 0
    assert (tmp_path / "benchmark_report.html").read_text().startswith("<!doctype html>")


def test_one_controller_only_says_it_cannot_compare():
    assert "Not enough" in headline(summarise({"batches": {}, "aborted": [], "errors": []}))


def test_headline_is_the_mean_of_the_per_case_changes(tmp_path):
    # An unbalanced session: tuned PX4 flown twice more in the loose wind case would tilt a
    # change of the pooled means; the headline averages the per-case changes instead.
    result = scored(tmp_path)
    tuned = result["batches"]["tuned_px4"]
    extra = [dict(r, seed=int(r["seed"]) + 10, p95_m=r["p95_m"] * 3.0)
             for r in tuned["rows"] if r["scenario"] == "wind_4ms"]
    tuned["rows"].extend(extra)
    agg = tuned["aggregate"]["wind_4ms"]
    rows = [r for r in tuned["rows"] if r["scenario"] == "wind_4ms"]
    agg["n"] = len(rows)
    agg["p95_m_mean"] = statistics.fmean(r["p95_m"] for r in rows)
    sm = summarise(result)
    expected = statistics.fmean(c["change"] for c in sm["cases"] if not math.isnan(c["change"]))
    assert sm["change"] == pytest.approx(expected) and sm["cases_compared"] == 2
    assert "the mean change over 2 cases" in headline(sm)


def test_strip_chart_survives_a_hold_point_metres_off():
    cases = [{"case": "calm", "tuned_px4": {"p95_runs": [(1, 42.0)]},
              "structure_hold": {"p95_runs": [(1, 0.2)]}}]
    svg = strip_chart(cases)
    assert svg.startswith("<svg") and "<circle" in svg and "<rect" in svg
