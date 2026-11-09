"""The SITL image's payload edit (sim2real/docker/add_payload.py) on an x500-shaped model."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "docker" / "add_payload.py"
_spec = importlib.util.spec_from_file_location("add_payload", SCRIPT)
add_payload = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(add_payload)

# The shape of PX4-gazebo-models x500/model.sdf: base merged in, then the motor plugins.
X500 = """<?xml version="1.0" encoding="UTF-8"?>
<sdf version='1.9'>
  <model name='x500'>
    <include merge='true'>
      <uri>model://x500_base</uri>
    </include>
    <plugin filename="gz-sim-multicopter-motor-model-system"
      name="gz::sim::systems::MulticopterMotorModel"><motorNumber>0</motorNumber></plugin>
  </model>
</sdf>
"""


def test_payload_is_a_fixed_link_under_base_link_with_box_inertia():
    out = add_payload.add_payload(X500, 1.032, 0.05)
    model = ET.fromstring(out.split("?>", 1)[1]).find("model")
    assert model.get("name") == "x500"  # same model, so x500_0 stays the instance name
    link = model.find("link[@name='structure_hold_payload']")
    assert link.find("pose").get("relative_to") == "base_link"
    assert [float(v) for v in link.find("pose").text.split()] == [0, 0, -0.05, 0, 0, 0]
    assert float(link.find("inertial/mass").text) == pytest.approx(1.032)
    ixx = float(link.find("inertial/inertia/ixx").text)
    assert ixx == pytest.approx(1.032 * (0.1 ** 2 + 0.05 ** 2) / 12, rel=1e-5)
    assert link.find("collision") is None  # inside the frame, above the legs
    joint = model.find("joint[@name='structure_hold_payload_joint']")
    assert joint.get("type") == "fixed"
    assert (joint.find("parent").text, joint.find("child").text) == (
        "base_link", "structure_hold_payload")
    # Everything that was there is still there, in order.
    assert model.find("include/uri").text == "model://x500_base"
    assert out.index("MulticopterMotorModel") < out.index("structure_hold_payload")


def test_payload_refuses_twice_and_bad_values(tmp_path):
    once = add_payload.add_payload(X500, 0.5, 0.05)
    with pytest.raises(ValueError):
        add_payload.add_payload(once, 0.5, 0.05)
    for mass, off in ((0.0, 0.05), (-1.0, 0.05), (0.5, -0.1)):
        with pytest.raises(ValueError):
            add_payload.add_payload(X500, mass, off)
    with pytest.raises(ValueError):
        add_payload.add_payload("<sdf/>", 0.5, 0.05)
    f = tmp_path / "model.sdf"
    f.write_text(X500)
    assert add_payload.main([str(f), "--mass-kg", "0.516"]) == 0
    assert "<mass>0.516</mass>" in f.read_text()
