"""structure_hold_params.yaml (PX4 v1.18's parameter format) matches structure_hold_params.c.

The C file stays the source of truth; the YAML is generated from it
(sim2real/px4/tools/params_yaml.py) and read only by PX4 trees that no longer read C
parameter files. A parameter added or changed in the C file must reach the YAML, or a
v1.18 build would fly with the old definition.
"""

from __future__ import annotations

import json

import yaml

from sim2real.px4.tests.test_param_limits import MODULE, _params_c
from sim2real.px4.tools import params_yaml


def test_yaml_is_regenerated_from_the_c_file() -> None:
    committed = (MODULE / "structure_hold_params.yaml").read_text()
    assert committed == params_yaml.render(params_yaml.build()), (
        "stale: run python -m sim2real.px4.tools.params_yaml")


def test_yaml_carries_every_parameter_with_the_same_numbers() -> None:
    # Read back through the limits test's own parser of the C file, not the generator's.
    groups = yaml.safe_load((MODULE / "structure_hold_params.yaml").read_text())["parameters"]
    definitions = {n: d for g in groups for n, d in g["definitions"].items()}
    params = _params_c()
    assert list(definitions) == list(params) and len(params) >= 25
    for name, p in params.items():
        d = definitions[name]
        assert float(d["default"]) == p["default"], name
        assert (d["type"] == "float") == (p["kind"] == "FLOAT"), name
        if d["type"] == "boolean":
            continue
        lo, hi = d.get("min"), d.get("max")
        if d["type"] == "enum" and lo is None:
            lo, hi = min(d["values"]), max(d["values"])
        assert (lo, hi) == (p.get("min"), p.get("max")), name
        assert d["description"]["short"], name


def test_module_hands_the_yaml_to_px4_only_where_the_c_file_is_not_read() -> None:
    cmake = (MODULE / "CMakeLists.txt").read_text()
    assert 'NOT sh_parameters_cmake MATCHES "PX4_MODULE_PATHS"' in cmake
    assert "MODULE_CONFIG structure_hold_params.yaml" in cmake
    assert "structure_hold_params.c" not in cmake.split("px4_add_module(")[1]


def _built_json(tmp_path, edit=None):
    """parameters.json as a PX4 build writes it, made from the C file's definitions."""
    groups = params_yaml.build()["parameters"]
    params = []
    for name, d in ((n, d) for g in groups for n, d in g["definitions"].items()):
        p = {"name": name, "type": "Float" if d["type"] == "float" else "Int32",
             "default": d["default"], "shortDesc": d["description"]["short"]}
        if "long" in d["description"]:
            p["longDesc"] = d["description"]["long"].replace("\n", " ")
        for ours, theirs in (("min", "min"), ("max", "max"), ("unit", "units"),
                             ("decimal", "decimalPlaces"), ("reboot_required", "rebootRequired")):
            if ours in d:
                p[theirs] = d[ours]
        if d["type"] == "enum":
            p["values"] = [{"value": v, "description": t} for v, t in d["values"].items()]
        params.append(p)
    params.append({"name": "MC_ROLL_P", "type": "Float", "default": 6.5})  # PX4's own: ignored
    if edit:
        edit({p["name"]: p for p in params}, params)
    path = tmp_path / "parameters.json"
    path.write_text(json.dumps({"parameters": params}))
    return path


def test_a_build_with_the_same_metadata_has_no_differences(tmp_path) -> None:
    assert params_yaml.differences(_built_json(tmp_path)) == []


def test_a_build_that_departs_from_the_c_file_is_reported(tmp_path) -> None:
    def edit(by_name, params):
        by_name["SH_WD_EST_TO"]["default"] = 0.5
        by_name["SH_FB_EN"]["values"].pop()
        params.remove(by_name["SH_EN"])
        params.append({"name": "SH_EXTRA", "type": "Int32", "default": 0})

    diffs = params_yaml.differences(_built_json(tmp_path, edit))
    assert len(diffs) == 4
    assert any(d.startswith("SH_WD_EST_TO.default") for d in diffs)
    assert any(d.startswith("SH_FB_EN.values") for d in diffs)
    assert "SH_EN: not in the build" in diffs
    assert any(d.startswith("SH_EXTRA: in the build") for d in diffs)
