"""One-page benchmark report from a partner's scored flight session (``docs/partner-pack.md``).

    python -m eval.partner_report <session>/scored/session_summary.json [--out report.html]
    python -m eval.partner_session <session>/session.yaml     # writes it too, next to report.md

``eval.partner_session`` scores the session and prints the engineer's tables. This turns the
same result into the page a drone maker shows their manager or safety reviewer: what was
flown, the headline (Structure Hold's 95th-percentile hold error against tuned PX4 on the
same drone, same day), wall contacts and holds that ended early, a per-case table, a
per-run chart, every handback in plain words, and how it was scored. One self-contained
HTML file (no scripts, no web fonts) that prints on one A4 or Letter page; print it to PDF
from any browser, or pass ``--pdf`` where Playwright is installed.

Nothing here scores anything: every number comes from ``session_summary.json``, so the page
and the tables can never disagree.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NAMES = {"tuned_px4": "Tuned PX4", "structure_hold": "Structure Hold"}
COLOURS = {"structure_hold": "var(--series-1)", "tuned_px4": "var(--series-2)"}
TRIP_TEXT = {
    None: "reason not logged",
    "none": "no watchdog trip (pilot or PX4 ended it)",
    "test_fault": "scripted test trip",
    "estimate_invalid": "position estimate lost",
    "estimate_stale": "position estimate late",
    "tilt": "tilt limit reached",
    "position_error": "position estimate jumped",
    "estimate_reset": "position estimate reset",
    "landed": "landed",
    "range_lost": "rangefinder lost",
    "estimate_frozen": "position estimate froze",
    "unknown": "unknown reason",
}
NAV_TEXT = {0: "Manual mode", 1: "Altitude mode", 2: "Position mode", 4: "Hold mode", 18: "Land",
            24: "Structure Land", 25: "Structure Hover", 23: "Structure Hold"}
ALIGN_WARN_M = 0.05  # an EKF-to-mocap fit worse than this makes the hold point suspect


def _num(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def fmt(x: Any, nd: int = 2, unit: str = "") -> str:
    v = _num(x)
    return "n/a" if math.isnan(v) else f"{v:.{nd}f}{unit}"


def summarise(result: dict[str, Any]) -> dict[str, Any]:
    """The numbers the page shows, from ``eval.partner_session.score_session``'s result."""
    b = result.get("batches", {})
    per: dict[str, Any] = {}
    for c in NAMES:
        rows = b.get(c, {}).get("rows", [])
        p95 = [r["p95_m"] for r in rows if not math.isnan(_num(r.get("p95_m")))]
        clear = [r["min_clearance_m"] for r in rows
                 if not math.isnan(_num(r.get("min_clearance_m")))]
        per[c] = {"n": len(rows), "p95_mean": statistics.fmean(p95) if p95 else math.nan,
                  "closest": min(clear) if clear else math.nan,
                  "contacts": sum(int(r.get("contacts", 0)) > 0 for r in rows),
                  "early": sum(bool(r.get("ended_early")) for r in rows)}
    cases = []
    for name in dict.fromkeys(r["scenario"] for bt in b.values() for r in bt["rows"]):
        row = {"case": name}
        for c in NAMES:
            agg = b.get(c, {}).get("aggregate", {}).get(name)
            rs = [r for r in b.get(c, {}).get("rows", []) if r["scenario"] == name]
            row[c] = None if agg is None else {
                "n": agg["n"], "p95_mean": agg["p95_m_mean"], "p95_worst": agg["p95_m_worst"],
                "toward_worst": agg["worst_toward_surface_m_worst"],
                "closest": agg["min_clearance_m_worst"],
                "contacts": sum(int(r.get("contacts", 0)) > 0 for r in rs),
                "early": sum(bool(r.get("ended_early")) for r in rs),
                "p95_runs": [(int(r["seed"]), _num(r["p95_m"])) for r in rs]}
        x, y = row["tuned_px4"], row["structure_hold"]
        row["change"] = (100.0 * (y["p95_mean"] - x["p95_mean"]) / x["p95_mean"]
                         if x and y and x["p95_mean"] and not math.isnan(x["p95_mean"])
                         else math.nan)
        cases.append(row)
    # The headline is the mean of the per-case changes (each case compares the two
    # controllers on the same conditions), not the change of the pooled means, which an
    # unbalanced session (one controller flown more often in the hard case) would tilt.
    changes = [c["change"] for c in cases if not math.isnan(c["change"])]
    change = statistics.fmean(changes) if changes else math.nan
    events = []
    for c in NAMES:
        for r in b.get(c, {}).get("rows", []):
            if r.get("ended_early"):
                nxt = NAV_TEXT.get(r.get("next_nav_state"),
                                   f"nav_state {r.get('next_nav_state')}")
                why = TRIP_TEXT.get(r.get("trip_reason"), str(r.get("trip_reason")))
                if c == "tuned_px4":
                    why = "failsafe" if r.get("failsafe") else "mode changed"
                held = r["window_s"][1] - r["window_s"][0]
                events.append(f"{NAMES[c]}, {r['scenario']} run {r['seed']} ({r['file']}): "
                              f"hold ended after {held:.0f} s, {why}; handed to {nxt}.")
            if _num(r.get("align_rms_m")) > ALIGN_WARN_M:
                events.append(f"{NAMES[c]}, {r['scenario']} run {r['seed']}: the EKF-to-mocap "
                              f"fit is off by {r['align_rms_m']:.2f} m RMS; its hold point is "
                              "uncertain by about that much.")
    for ab in result.get("aborted", []):
        who = NAMES.get(ab.get("controller"), ab.get("controller"))
        events.append(f"Not flown or aborted: {ab.get('file')} ({who}, {ab.get('case')} run "
                      f"{ab.get('run')}): {ab.get('reason') or 'no reason given'}.")
    for e in result.get("errors", []):
        events.append(f"Not scored: {e.get('file')}: {e.get('error')}.")
    return {"per": per, "change": change, "cases": cases, "events": events,
            "cases_compared": len(changes)}


def headline(sm: dict[str, Any]) -> str:
    ch, per = sm["change"], sm["per"]
    if math.isnan(ch):
        return "Not enough scored runs of both controllers to compare."
    word = "tighter" if ch < 0 else "looser"
    n = per["structure_hold"]["n"], per["tuned_px4"]["n"]
    k = sm.get("cases_compared", 0)
    return (f"Structure Hold held {abs(ch):.0f} % {word} than tuned PX4 on the same drone "
            f"(95th-percentile distance from the hold point, the mean change over {k} "
            f"case{'s' if k != 1 else ''}, {n[0]} and {n[1]} scored holds).")


def strip_chart(cases: list[dict[str, Any]]) -> str:
    """Inline SVG: one row per case, every run's p95 as a dot (Structure Hold circles, tuned
    PX4 squares), on one shared metre axis from 0."""
    vals = [v for row in cases for c in NAMES if row[c] for _, v in row[c]["p95_runs"]
            if not math.isnan(v)]
    if not vals:
        return ""
    top = max(vals) * 1.1
    step = next((s for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0) if top / s <= 6),
                None)
    if step is None:  # a hold point metres off (a bad mocap fit): a coarse decade axis
        step = 10.0 ** math.ceil(math.log10(top / 6))
    top = math.ceil(top / step) * step
    left, right, row_h, pad = 88, 16, 30, 26
    width = 640
    plot_w = width - left - right
    height = pad + row_h * len(cases) + 22

    def x(v: float) -> float:
        return left + plot_w * v / top

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="p95 per run, by case"'
           ' class="chart">']
    k = 0.0
    while k <= top + 1e-9:
        out.append(f'<line x1="{x(k):.1f}" x2="{x(k):.1f}" y1="{pad - 6}" '
                   f'y2="{height - 22}" class="grid"/>')
        out.append(f'<text x="{x(k):.1f}" y="{height - 6}" class="tick" '
                   f'text-anchor="middle">{k:g}</text>')
        k += step
    out.append(f'<text x="{width - right}" y="{pad - 12}" class="tick" text-anchor="end">'
               "p95 distance from hold point, m</text>")
    for i, row in enumerate(cases):
        cy = pad + row_h * i + row_h / 2
        out.append(f'<text x="{left - 10}" y="{cy + 4:.1f}" class="label" text-anchor="end">'
                   f"{html.escape(str(row['case']))}</text>")
        for j, c in enumerate(("structure_hold", "tuned_px4")):
            if not row[c]:
                continue
            yy = cy + (-5 if j == 0 else 5)
            for run, v in row[c]["p95_runs"]:
                if math.isnan(v):
                    continue
                tip = f"{NAMES[c]}, {row['case']} run {run}: p95 {v:.3f} m"
                if c == "structure_hold":
                    mark = (f'<circle cx="{x(v):.1f}" cy="{yy:.1f}" r="4.5" '
                            f'fill="{COLOURS[c]}" class="dot"/>')
                else:
                    mark = (f'<rect x="{x(v) - 4:.1f}" y="{yy - 4:.1f}" width="8" height="8" '
                            f'rx="1.5" fill="{COLOURS[c]}" class="dot"/>')
                out.append(f"<g><title>{html.escape(tip)}</title>{mark}</g>")
    out.append("</svg>")
    return "".join(out)


CSS = """
:root{--surface:#fff;--surface-2:#f5f4f0;--text:#1d1d1b;--text-2:#55554f;--muted:#8a8a82;
--rule:#e2e1dc;--series-1:#2a78d6;--series-2:#eb6834;--good:#1a7f37;--bad:#c62828}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;
--surface-2:#242422;--text:#f0efec;--text-2:#c3c2b7;--muted:#8f8e86;--rule:#3a3a37;
--series-1:#3987e5;--series-2:#d95926;--good:#4cc26b;--bad:#ef6b6b}}
:root[data-theme="dark"]{--surface:#1a1a19;--surface-2:#242422;--text:#f0efec;
--text-2:#c3c2b7;--muted:#8f8e86;--rule:#3a3a37;--series-1:#3987e5;--series-2:#d95926;
--good:#4cc26b;--bad:#ef6b6b}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text);
font:13px/1.45 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px}
h1{font-size:20px;margin:0 0 2px}h2{font-size:13px;margin:16px 0 6px;text-transform:uppercase;
letter-spacing:.04em;color:var(--text-2)}
.sub{color:var(--text-2);margin:0 0 12px}
.head{font-size:15px;font-weight:600;margin:10px 0 12px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.tile{background:var(--surface-2);border-radius:6px;padding:8px 10px}
.tile b{display:block;font-size:18px}.tile span{color:var(--text-2);font-size:11px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:4px 6px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;color:var(--text-2);font-size:11px}
td.n{text-align:right}th.n{text-align:right}
.key{display:inline-block;width:9px;height:9px;margin-right:5px;vertical-align:baseline}
.chart{width:100%;height:auto}.chart .grid{stroke:var(--rule)}
.chart .tick{fill:var(--muted);font-size:10px}.chart .label{fill:var(--text-2);font-size:11px}
.chart .dot{stroke:var(--surface);stroke-width:1.5}
.legend{color:var(--text-2);font-size:11px;margin:2px 0 0}
ul{margin:0;padding-left:18px}li{margin:2px 0}
.setup{display:grid;grid-template-columns:repeat(2,1fr);gap:2px 16px;color:var(--text-2);
font-size:11.5px}.setup b{color:var(--text);font-weight:600}
footer{margin-top:14px;color:var(--muted);font-size:10.5px;border-top:1px solid var(--rule);
padding-top:6px}
@media (max-width:560px){.tiles{grid-template-columns:repeat(2,1fr)}
.setup{grid-template-columns:1fr}table{font-size:11.5px}}
@media print{:root{--surface:#fff;--surface-2:#f5f4f0;--text:#1d1d1b;--text-2:#55554f;
--muted:#8a8a82;--rule:#e2e1dc;--series-1:#2a78d6;--series-2:#eb6834}
@page{size:A4;margin:12mm}main{padding:0;max-width:none}body{font-size:11.5px}}
"""


def render(result: dict[str, Any], commit: str = "unknown", generated: str | None = None
           ) -> str:
    s = result.get("session", {})
    sm = summarise(result)
    per = sm["per"]
    e = html.escape
    generated = generated or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def tile(value: str, label: str) -> str:
        return f'<div class="tile"><b>{e(value)}</b><span>{e(label)}</span></div>'

    def pair(key: str, nd: int = 2) -> str:
        return f"{fmt(per['structure_hold'][key], nd)} vs {fmt(per['tuned_px4'][key], nd)}"

    ch = sm["change"]
    tiles = [
        tile("n/a" if math.isnan(ch) else f"{ch:+.0f} %", "p95 hold error, Structure Hold vs "
             "tuned PX4"),
        tile(f"{per['structure_hold']['contacts']} vs {per['tuned_px4']['contacts']}",
             "holds with a wall contact (Structure Hold vs tuned PX4)"),
        tile(f"{per['structure_hold']['early']} vs {per['tuned_px4']['early']}",
             "holds that ended early"),
        tile(pair("closest") + " m", "closest to the wall, any hold"),
    ]
    rows = []
    for row in sm["cases"]:
        for i, c in enumerate(("structure_hold", "tuned_px4")):
            r = row[c]
            case = e(str(row["case"])) if i == 0 else ""
            change = ("" if i else ("n/a" if math.isnan(row["change"])
                                    else f"{row['change']:+.0f} %"))
            key = f'<span class="key" style="background:{COLOURS[c]}"></span>'
            if r is None:
                rows.append(f"<tr><td>{case}</td><td>{key}{NAMES[c]}</td>"
                            '<td colspan="6">not flown</td></tr>')
                continue
            rows.append(
                f"<tr><td>{case}</td><td>{key}{NAMES[c]}</td><td class='n'>{r['n']}</td>"
                f"<td class='n'>{fmt(r['p95_mean'], 3)} ({fmt(r['p95_worst'], 3)})</td>"
                f"<td class='n'>{fmt(r['toward_worst'])}</td>"
                f"<td class='n'>{fmt(r['closest'])}</td>"
                f"<td class='n'>{r['contacts']} / {r['early']}</td>"
                f"<td class='n'>{change}</td></tr>")
    events = "".join(f"<li>{e(t)}</li>" for t in sm["events"]) or \
        "<li>Every hold ran its full length; nothing was aborted or left unscored.</li>"
    wall = s.get("wall") or {}
    setup = [
        ("Drone", s.get("drone")), ("Site", s.get("site")), ("Pilot", s.get("pilot")),
        ("Firmware", s.get("firmware")), ("Policy", s.get("policy")),
        ("Position source", s.get("vio")), ("Truth", s.get("truth", "mocap")),
        ("Tuned PX4", f"{s.get('tuned_px4_params', 'n/a')}, "
                      f"{'Position' if s.get('baseline_mode') == 'position' else 'Hold'} mode"),
        ("Structure Hold params", s.get("structure_hold_params")),
        ("Hold", f"{fmt(s.get('hold_s', 60), 0)} s, first {fmt(s.get('settle_s', 2), 0)} s "
                 "not scored"),
        ("Contact", f"closer than {fmt(s.get('contact_radius_m'), 2)} m to the wall"
         if s.get("contact_radius_m") else "default prop reach"),
        ("Wall", "measured" if wall else "not given (no wall metrics)"),
    ]
    setup_html = "".join(f"<div><b>{e(k)}:</b> {e(str(v if v is not None else 'n/a'))}</div>"
                         for k, v in setup)
    truth = ("real flights, scored on motion-capture truth" if s.get("truth", "mocap") == "mocap"
             else f"scored on {s.get('truth')} truth")
    mocap = s.get("truth", "mocap") == "mocap"
    truth_src = "from motion capture" if mocap else f"from {s.get('truth')}"
    carried = (", carried into the mocap frame by fitting the drone's estimate to mocap over "
               "the take-off" if mocap else "")
    title = f"Structure Hold vs tuned PX4 on {s.get('drone', 'a partner drone')}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Benchmark report</title><style>{CSS}</style></head>
<body><main>
<h1>{e(title)}</h1>
<p class="sub">{e(str(s.get('site', '')))} · {e(str(s.get('date', '')))} · {e(truth)}</p>
<p class="head">{e(headline(sm))}</p>
<div class="tiles">{''.join(tiles)}</div>
<h2>Per case</h2>
<table><thead><tr><th>Case</th><th>Controller</th><th class="n">Holds</th>
<th class="n">p95 mean (worst), m</th><th class="n">Worst toward wall, m</th>
<th class="n">Closest to wall, m</th><th class="n">Contacts / ended early</th>
<th class="n">p95 change</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Every hold</h2>
{strip_chart(sm['cases'])}
<p class="legend"><span class="key" style="background:var(--series-1);border-radius:50%">
</span>Structure Hold (circles) &nbsp; <span class="key" style="background:var(--series-2)">
</span>Tuned PX4 (squares). Lower is tighter.</p>
<h2>Handbacks and exceptions</h2>
<ul>{events}</ul>
<h2>Set-up and method</h2>
<div class="setup">{setup_html}</div>
<p class="legend">Each hold is scored on the true position {e(truth_src)}, never on the
drone's own estimate, with the same code as our simulator and Gazebo benchmark. The hold point
is the controller's setpoint at the start of the hold{e(carried)}. p95 is the
95th-percentile distance from that point over the hold. A contact is any moment closer to
the wall than the prop reach; a hold that ended before its full length counts as ended
early, whatever the reason. The two controllers flew alternately on the same drone, the
same day.</p>
<footer>Confidential. Generated {e(generated)} by eval/partner_report.py at commit
{e(commit)} from session_summary.json (eval/partner_session.py).</footer>
</main></body></html>
"""


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def write_pdf(html_path: Path, pdf_path: Path) -> None:
    from playwright.sync_api import sync_playwright  # optional; only for --pdf

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri())
        page.emulate_media(media="print")
        page.pdf(path=str(pdf_path), format="A4", print_background=True)
        browser.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("summary", type=Path, help="session_summary.json from eval.partner_session")
    ap.add_argument("--out", type=Path, help="HTML file (default: benchmark_report.html "
                                             "next to the summary)")
    ap.add_argument("--pdf", action="store_true", help="also print it to PDF (needs Playwright)")
    args = ap.parse_args(argv)
    result = json.loads(args.summary.read_text())
    out = args.out or args.summary.parent / "benchmark_report.html"
    out.write_text(render(result, git_commit()))
    print("report:", out)
    if args.pdf:
        write_pdf(out, out.with_suffix(".pdf"))
        print("pdf:", out.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
