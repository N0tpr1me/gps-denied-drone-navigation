"""The compiled-in policy must be the released one: regenerating the header from the policy
it names (``NAME`` in PolicyWeights.hpp, training/policies/<name>.onnx) has to reproduce the
committed file byte for byte. Needs onnx (the ``torch`` extra); skipped without it."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")

from sim2real.px4.tools.export_policy_header import (  # noqa: E402
    DEFAULT_OUT,
    UnsupportedGraph,
    forward,
    generate,
    parse_mlp,
)

REPO = Path(__file__).resolve().parents[3]
COMMITTED = DEFAULT_OUT.read_text()
NAME = re.search(r'NAME = "([^"]+)"', COMMITTED).group(1)
POLICY = REPO / "training" / "policies" / f"{NAME}.onnx"


def test_committed_header_matches_released_policy(tmp_path: Path):
    text = generate(POLICY, tmp_path / "PolicyWeights.hpp")
    # No `assert a == b`: pytest would diff two 300 KB strings for minutes on a mismatch.
    if text != COMMITTED:
        pytest.fail("sim2real/px4/structure_hold/PolicyWeights.hpp is out of date: run "
                    f"python -m sim2real.px4.tools.export_policy_header {POLICY.relative_to(REPO)}")


def test_parsed_mlp_matches_onnxruntime():
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    net = parse_mlp(POLICY)
    sess = ort.InferenceSession(str(POLICY), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal((32, net["in_size"])).astype(np.float32)
    y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    assert np.abs(forward(net, x) - y).max() < 1e-4


def test_unsupported_graph_is_refused(tmp_path: Path):
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [None, 2])
    y = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [None, 2])
    node = helper.make_node("Sigmoid", ["obs"], ["actions"])
    model = helper.make_model(helper.make_graph([node], "g", [x], [y]))
    path = tmp_path / "bad.onnx"
    onnx.save(model, str(path))
    with pytest.raises(UnsupportedGraph):
        parse_mlp(path)
