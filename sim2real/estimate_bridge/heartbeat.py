"""Decode PX4's HEARTBEAT custom_mode, without pymavlink or gz (host and CI can import it)."""

from __future__ import annotations

# PX4 custom modes (px4_custom_mode.h), as sim2real/px4/sitl_test.py decodes the heartbeat.
MAIN_AUTO, SUB_EXTERNAL1 = 4, 11


def structure_hold_engaged(custom_mode: int) -> bool:
    """Whether a PX4 HEARTBEAT custom_mode says Structure Hold (the first external mode) is
    the active mode."""
    main_mode = (custom_mode >> 16) & 0xFF
    sub_mode = (custom_mode >> 24) & 0xFF
    return main_mode == MAIN_AUTO and sub_mode >= SUB_EXTERNAL1


# The module refuses to engage on the ground, and the flight sequence tries exactly that
# before take-off (sim2real/px4/sitl_test.py step 0, PR #101), so the mode shows on the
# heartbeat for a moment while the vehicle sits on the ground. Before this check the bridge
# anchored an engage-keyed fault schedule on that refusal, 10 to 15 s before the real
# engage. An engage counts for the anchor only this far above the bridge's first sample.
ENGAGE_MIN_HEIGHT_M = 0.5


def airborne_engage(engaged: bool, z_m: float, first_z_m: float) -> bool:
    """Whether a sample counts as Structure Hold flying for the bridge's engage anchor: the
    mode on the heartbeat while the vehicle is clear of the ground (truth height ``z_m``,
    against ``first_z_m``, where the bridge's first truth sample put it)."""
    return bool(engaged) and z_m - first_z_m >= ENGAGE_MIN_HEIGHT_M
