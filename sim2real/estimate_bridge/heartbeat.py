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
