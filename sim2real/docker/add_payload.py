"""Fix a payload under the x500 in its Gazebo model, before PX4 spawns it.

    python3 add_payload.py MODEL_SDF --mass-kg 1.03 [--offset-down-m 0.05]

The entrypoint (``entrypoint.sh``) runs this when ``STRUCTURE_HOLD_PAYLOAD_KG`` is set, so a
scenario's ``payload`` block (``sim2real/gazebo/scenario.py``) flies a heavier x500 with the
same model name (``x500_0``), which the wind plugin, the estimate bridge and the scorers look
up. Each flight is its own ``docker run --rm``, so the edited model dies with the container.

The payload is a rigid link on a fixed joint to ``base_link``: a 0.10 x 0.10 x 0.05 m box of
uniform density whose centre sits ``offset_down_m`` below base_link, with no collision shape
(it sits inside the frame, above the landing gear). Gazebo then carries its mass and inertia
in the dynamics, so PX4 sees a heavier, slower drone, not only a weight. PX4's hover thrust
estimator learns the new hover thrust in flight, as it would on a real drone with a payload.
Standard library only: it runs on the image's system Python.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "<!-- structure_hold payload -->"
BOX_M = (0.10, 0.10, 0.05)  # x, y, z


def box_inertia(mass: float, size: tuple[float, float, float] = BOX_M) -> tuple[float, ...]:
    """Diagonal inertia (ixx, iyy, izz) of a uniform box about its centre."""
    x, y, z = size
    return (mass * (y * y + z * z) / 12.0, mass * (x * x + z * z) / 12.0,
            mass * (x * x + y * y) / 12.0)


def payload_sdf(mass: float, offset_down_m: float) -> str:
    ixx, iyy, izz = box_inertia(mass)
    return f"""    {MARKER}
    <link name="structure_hold_payload">
      <pose relative_to="base_link">0 0 {-offset_down_m:.6g} 0 0 0</pose>
      <inertial>
        <mass>{mass:.6g}</mass>
        <inertia>
          <ixx>{ixx:.6g}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{iyy:.6g}</iyy><iyz>0</iyz>
          <izz>{izz:.6g}</izz>
        </inertia>
      </inertial>
      <visual name="payload_visual">
        <geometry><box><size>{BOX_M[0]} {BOX_M[1]} {BOX_M[2]}</size></box></geometry>
      </visual>
    </link>
    <joint name="structure_hold_payload_joint" type="fixed">
      <parent>base_link</parent>
      <child>structure_hold_payload</child>
    </joint>
"""


def add_payload(sdf: str, mass: float, offset_down_m: float) -> str:
    """``sdf`` with the payload inserted before the model's closing tag. Refuses a model that
    already has one, so a payload is never added twice."""
    if mass <= 0 or offset_down_m < 0:
        raise ValueError("payload needs mass > 0 and offset >= 0")
    if MARKER in sdf:
        raise ValueError("the model already carries a payload")
    end = sdf.rfind("</model>")
    if end < 0:
        raise ValueError("no </model> in the SDF")
    return sdf[:end] + payload_sdf(mass, offset_down_m) + "  " + sdf[end:]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_sdf", type=Path)
    ap.add_argument("--mass-kg", type=float, required=True)
    ap.add_argument("--offset-down-m", type=float, default=0.05)
    args = ap.parse_args(argv)
    text = add_payload(args.model_sdf.read_text(), args.mass_kg, args.offset_down_m)
    args.model_sdf.write_text(text)
    print(f"payload: {args.mass_kg} kg, {args.offset_down_m} m below base_link, "
          f"in {args.model_sdf}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
