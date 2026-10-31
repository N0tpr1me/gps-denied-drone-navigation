"""Turn a released ``policy.onnx`` + ``policy.json`` into a C++ header the PX4 module compiles in.

The onboard inference path is deliberately not an ONNX runtime (docs/decisions.md): the
released actors are tiny MLPs (``Sub``, ``Div``, ``Gemm``/``Elu`` chains, a few tens of
kilobytes), and a flight controller build (NuttX or a companion Linux target) wants no heap,
no dynamic loading and no third-party library. This script reads the ONNX graph, checks it
is exactly that shape, and writes the weights (shortest round-trip float32 literals) and the
``policy.json`` contract (history length, layout, action scaling, the mock airframe, who
owns yaw, whether the fallback flies by default) into
``structure_hold/PolicyWeights.hpp``, which ``MlpPolicy.hpp`` evaluates.

Run it after every policy release and commit the header; ``tests/test_policy_header.py``
fails when the committed header no longer matches the released policy.

    python -m sim2real.px4.tools.export_policy_header training/policies/structure_hold_v2.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "structure_hold" / "PolicyWeights.hpp"
SUPPORTED_ACTIVATIONS = {"Elu", "Relu", "Tanh"}


class UnsupportedGraph(ValueError):
    pass


def _initializers(model) -> dict[str, np.ndarray]:
    from onnx import numpy_helper

    return {t.name: numpy_helper.to_array(t).astype(np.float32) for t in model.graph.initializer}


def parse_mlp(onnx_path: Path) -> dict:
    """Read the ONNX file into ``{"mean", "std", "layers": [(W, b, activation|None)]}``.

    Accepted graph: optional ``Sub`` (mean) then ``Div`` (std) on the input, then a chain of
    ``Gemm`` (transB=1, alpha=beta=1) each followed by one activation except the last.
    Anything else raises ``UnsupportedGraph`` so a changed export is noticed, not silently
    mis-run on the vehicle.
    """
    import onnx

    model = onnx.load(str(onnx_path))
    init = _initializers(model)
    nodes = list(model.graph.node)
    (inp,) = model.graph.input
    (out,) = model.graph.output
    in_size = int(inp.type.tensor_type.shape.dim[-1].dim_value)
    current = inp.name
    i = 0
    mean = np.zeros(in_size, np.float32)
    std = np.ones(in_size, np.float32)

    def take(op: str):
        nonlocal i, current
        if i >= len(nodes) or nodes[i].op_type != op or nodes[i].input[0] != current:
            return None
        node = nodes[i]
        i += 1
        current = node.output[0]
        return node

    if (node := take("Sub")) is not None:
        mean = init[node.input[1]].reshape(-1)
    if (node := take("Div")) is not None:
        std = init[node.input[1]].reshape(-1)
    if mean.shape != (in_size,) or std.shape != (in_size,):
        raise UnsupportedGraph("normaliser shape does not match the input size")

    layers: list[tuple[np.ndarray, np.ndarray, str | None]] = []
    while i < len(nodes):
        node = take("Gemm")
        if node is None:
            raise UnsupportedGraph(f"expected Gemm, found {nodes[i].op_type} at node {i}")
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get("transA", 0) != 0 or attrs.get("transB", 0) != 1 \
                or attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0:
            raise UnsupportedGraph(f"Gemm attributes not supported: {attrs}")
        w, b = init[node.input[1]], init[node.input[2]]
        activation = None
        if i < len(nodes):
            act = nodes[i]
            if act.op_type not in SUPPORTED_ACTIVATIONS or act.input[0] != current:
                raise UnsupportedGraph(f"unsupported activation {act.op_type} after Gemm")
            if act.op_type == "Elu":
                alpha = {a.name: onnx.helper.get_attribute_value(a) for a in act.attribute}
                if alpha.get("alpha", 1.0) != 1.0:
                    raise UnsupportedGraph("Elu alpha must be 1")
            activation = act.op_type
            i += 1
            current = act.output[0]
        layers.append((w, b, activation))
    if current != out.name:
        raise UnsupportedGraph("graph output is not the last layer's output")
    if not layers or layers[-1][2] is not None:
        raise UnsupportedGraph("the last layer must be a plain Gemm")
    prev = in_size
    for w, b, _ in layers:
        if w.shape[1] != prev or b.shape != (w.shape[0],):
            raise UnsupportedGraph(f"layer shape mismatch: {w.shape} after width {prev}")
        prev = w.shape[0]
    return {"in_size": in_size, "out_size": prev, "mean": mean, "std": std, "layers": layers}


def forward(net: dict, x: np.ndarray) -> np.ndarray:
    """Reference evaluation of the parsed MLP in float32 (what MlpPolicy.hpp computes)."""
    h = ((x.astype(np.float32) - net["mean"]) / net["std"]).astype(np.float32)
    for w, b, act in net["layers"]:
        h = (h @ w.T + b).astype(np.float32)
        if act == "Elu":
            h = np.where(h > 0, h, np.expm1(np.minimum(h, 0))).astype(np.float32)
        elif act == "Relu":
            h = np.maximum(h, 0)
        elif act == "Tanh":
            h = np.tanh(h).astype(np.float32)
    return h


def _literal(v: np.float32) -> str:
    """Shortest decimal literal that reads back as exactly this float32 (hexfloat literals
    are C++17 and PX4 compiles with gnu++14)."""
    text = str(np.float32(v))
    if "." not in text and "e" not in text and "n" not in text:
        text += ".0"
    return text + "f"


def _hexfloats(a: np.ndarray, per_line: int = 8) -> str:
    vals = [_literal(v) for v in a.reshape(-1).astype(np.float32)]
    lines = []
    for k in range(0, len(vals), per_line):
        lines.append("\t" + ", ".join(vals[k:k + per_line]) + ",")
    return "\n".join(lines)


def render_header(net: dict, meta: dict, onnx_path: Path, json_path: Path, sha: str) -> str:
    layout = meta["input"]["layout"]
    env = meta["env_cfg"]
    history = int(meta["input"]["history"])
    num_obs = int(meta["input"]["num_obs"])
    if history * num_obs != net["in_size"]:
        raise UnsupportedGraph(f"policy.json says {history} x {num_obs} inputs, "
                               f"the graph takes {net['in_size']}")
    if int(meta["output"]["size"]) != net["out_size"]:
        raise UnsupportedGraph("policy.json output size differs from the graph")
    act_map = {None: "Activation::None", "Elu": "Activation::Elu", "Relu": "Activation::Relu",
               "Tanh": "Activation::Tanh"}
    widths = [net["in_size"]] + [w.shape[0] for w, _, _ in net["layers"]]
    out = []
    out.append("// GENERATED by sim2real/px4/tools/export_policy_header.py. Do not edit.")
    out.append(f"// Source: training/policies/{onnx_path.name} (sha256 {sha[:16]}) "
               f"+ {json_path.name}")
    out.append("#pragma once")
    out.append("")
    out.append("#include \"MlpPolicy.hpp\"")
    out.append("")
    out.append("namespace structure_hold { namespace policy")
    out.append("{")
    out.append("")
    out.append(f"static constexpr const char *NAME = \"{onnx_path.stem}\";")
    out.append(f"static constexpr const char *SHA256_PREFIX = \"{sha[:16]}\";")
    out.append(f"static constexpr int HISTORY = {history};  // from policy.json input.history")
    out.append(f"static constexpr int NUM_OBS = {num_obs};")
    out.append(f"static constexpr int INPUT_SIZE = {net['in_size']};")
    out.append(f"static constexpr int OUTPUT_SIZE = {net['out_size']};")
    out.append(f"static constexpr int NUM_LAYERS = {len(net['layers'])};")
    out.append(f"static constexpr int MAX_WIDTH = {max(widths)};")
    out.append(f"static constexpr int RATE_HZ = {int(meta['rate_hz'])};")
    out.append("")
    out.append("// Observation layout (start index of each block in one NUM_OBS slice).")
    for name, (a, b) in layout.items():
        out.append(f"static constexpr int OBS_{name.upper()} = {a};  // size {b - a}")
    clip = float(meta["input"]["obs_pos_clip_m"])
    out.append(f"static constexpr float OBS_POS_CLIP_M = {clip!r}f;")
    # The engage-time input (v22 on): seconds since engage over ENGAGE_OBS_CAP_S, capped at
    # 1. Policies without it get the constants too, so the module code compiles either way.
    has_engage = "engage_time" in layout
    if not has_engage:
        out.append("static constexpr int OBS_ENGAGE_TIME = 0;  // unused: no engage-time input")
    cap = float(env.get("engage_obs_cap_s", 4.0))
    out.append(f"static constexpr bool HAS_ENGAGE_TIME = {str(has_engage).lower()};")
    out.append(f"static constexpr float ENGAGE_OBS_CAP_S = {cap!r}f;")
    out.append("")
    out.append("// Action scaling and the airframe the policy was trained on (mock sim).")
    max_rate = float(meta["output"]["max_rate_rad_s"])
    max_thrust = float(meta["output"]["max_thrust_n"])
    out.append(f"static constexpr float MAX_RATE_RAD_S = {max_rate!r}f;")
    out.append(f"static constexpr float MAX_THRUST_N = {max_thrust!r}f;")
    out.append(f"static constexpr float MASS_KG = {float(env['quad']['mass'])!r}f;")
    # policy.json output.yaw: "policy" when the policy was trained to hold the heading,
    # "module" (or absent, as for v2) when the module's yaw guard must own the heading.
    yaw_owner = meta["output"].get("yaw", "module")
    if yaw_owner not in ("policy", "module"):
        raise UnsupportedGraph(f"policy.json output.yaw must be 'policy' or 'module', "
                               f"got {yaw_owner!r}")
    out.append("// Whether the policy was trained to hold the heading (policy.json output.yaw).")
    from_policy = "true" if yaw_owner == "policy" else "false"
    out.append(f"static constexpr bool YAW_FROM_POLICY = {from_policy};")
    # policy.json output.fallback: "off" when the policy handles dropouts itself and the
    # brake-and-hover rule costs it (v3), "on" (or absent, as for v2) when the rule helps.
    fallback = meta["output"].get("fallback", "on")
    if fallback not in ("on", "off"):
        raise UnsupportedGraph(f"policy.json output.fallback must be 'on' or 'off', "
                               f"got {fallback!r}")
    out.append("// Whether the brake-and-hover fallback flies by default "
               "(policy.json output.fallback).")
    fallback_on = "true" if fallback == "on" else "false"
    out.append(f"static constexpr bool FALLBACK_DEFAULT_ON = {fallback_on};")
    out.append("")
    out.append(f"static constexpr float MEAN[INPUT_SIZE] = {{\n{_hexfloats(net['mean'])}\n}};")
    out.append(f"static constexpr float STD[INPUT_SIZE] = {{\n{_hexfloats(net['std'])}\n}};")
    for k, (w, b, _) in enumerate(net["layers"]):
        out.append(f"static constexpr float W{k}[{w.shape[0]} * {w.shape[1]}] = {{"
                   "  // row-major (out, in)")
        out.append(_hexfloats(w))
        out.append("};")
        out.append(f"static constexpr float B{k}[{b.shape[0]}] = {{\n{_hexfloats(b)}\n}};")
    out.append("")
    out.append("static constexpr Layer LAYERS[NUM_LAYERS] = {")
    for k, (w, _, act) in enumerate(net["layers"]):
        out.append(f"\t{{{w.shape[1]}, {w.shape[0]}, W{k}, B{k}, {act_map[act]}}},")
    out.append("};")
    out.append("")
    out.append("static constexpr Net NET{INPUT_SIZE, OUTPUT_SIZE, MEAN, STD, LAYERS, NUM_LAYERS};")
    out.append("")
    out.append("}} // namespace structure_hold::policy")
    out.append("")
    return "\n".join(out)


def generate(onnx_path: Path, out_path: Path = DEFAULT_OUT) -> str:
    json_path = onnx_path.with_suffix(".json")
    meta = json.loads(json_path.read_text())
    net = parse_mlp(onnx_path)
    sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    text = render_header(net, meta, onnx_path, json_path, sha)
    out_path.write_text(text)
    return text


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("onnx", type=Path, help="policy.onnx; policy.json must sit next to it")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    generate(args.onnx, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
