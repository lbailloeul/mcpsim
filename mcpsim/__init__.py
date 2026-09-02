"""mcpsim — millicharged-particle production & sensitivity CLI.

A config-driven front end over the existing DarkQuest / SHiP / LANL physics
scripts. The user sets beam energy, collision type, target and detector size and
gets a production plot plus a limit (exclusion) plot, combining four channels:
proton bremsstrahlung, meson 2-body decay, meson 3-body (Dalitz) decay, and
Drell-Yan.
"""

__version__ = "0.2.0"

from .config import Config, load_config, load_preset, list_presets

__all__ = ["Config", "load_config", "load_preset", "list_presets", "__version__"]
