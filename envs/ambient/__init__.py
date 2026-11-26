"""Air density, the battery, rain and rotors near a surface (``core``; NumPy only).

    from envs.ambient import AmbientConfig, sample_ambient
    p = sample_ambient(AmbientConfig(temperature_c=(35, 40), altitude_m=2000), 1024, seed=0)
    p.power           # thrust per command against the standard day, (N,)

The whole list of environmental factors, and where each one lives, is
``docs/environment-factors.md``.
"""

from envs.ambient.core import (
    AmbientConfig,
    AmbientParams,
    air_density,
    density_altitude_m,
    describe,
    hover_headroom,
    nearness,
    sag_gain,
    sample_ambient,
    surface_gain,
)

__all__ = ["AmbientConfig", "AmbientParams", "air_density", "density_altitude_m", "describe",
           "hover_headroom", "nearness", "sag_gain", "sample_ambient", "surface_gain"]
