"""Filesystem locations.

mcpsim lives in its own project directory and treats the original
``github-repo-scripts`` tree as a read-only *source repo* for data files,
precomputed bremsstrahlung grids, experiment contours, and the C++ / VEGAS
generators. Nothing here writes into the source repo; all generated output goes
under this project's ``mcpsim-output/``.

The source repo is located via, in order:
  1. the ``MCPSIM_SOURCE_REPO`` environment variable,
  2. a ``source_repo`` value injected from a config (set_source_repo),
  3. a ``github-repo-scripts`` sibling of this project directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
PRESETS_DIR = PACKAGE_DIR / "presets"

# Output (always inside this project, never the source repo).
WORK_DIR = PROJECT_ROOT / "mcpsim-output"
CACHE_DIR = WORK_DIR / "cache"

_DEFAULT_SOURCE = PROJECT_ROOT.parent / "github-repo-scripts"
_source_override: Optional[Path] = None


def set_source_repo(path: str | Path) -> None:
    """Override the source-repo location (e.g. from a config field)."""
    global _source_override
    _source_override = Path(path).expanduser().resolve()


def source_repo() -> Path:
    """Resolve the github-repo-scripts source tree."""
    if _source_override is not None:
        return _source_override
    env = os.environ.get("MCPSIM_SOURCE_REPO")
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_SOURCE


def require_source_repo() -> Path:
    repo = source_repo()
    if not repo.exists():
        raise FileNotFoundError(
            f"Source repo not found at {repo}. Set MCPSIM_SOURCE_REPO or the "
            f"'source_repo' config field to your github-repo-scripts checkout."
        )
    return repo


# -- derived locations (functions so a late source-repo override is honoured) --
def data_dir() -> Path:
    return source_repo() / "data-backup"


def contours_dir() -> Path:
    return source_repo() / "experiment-contours-small"


def brem_darkquest_dir() -> Path:
    return source_repo() / "DarkQuest-brem-backup"


def brem_ship_dir() -> Path:
    return source_repo() / "SHiP-brem-backup"


def mesongen_dir() -> Path:
    return source_repo() / "mesongen-backup"


def decay_dir() -> Path:
    return source_repo() / "decay-backup"


def lanl_dir() -> Path:
    return source_repo() / "lanl_12bar_pipeline"


def mcp_brem_script() -> Path:
    return data_dir() / "mCP_brem.py"


def resolve_data(path_like: str | Path) -> Path:
    """Resolve a config-supplied data path.

    Absolute paths are used as-is; bare names are looked up first in the source
    repo's data-backup, then relative to the source-repo root (so presets can
    reference e.g. ``DarkQuest-brem-backup`` or ``mship_values.txt`` by name).
    """
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    candidate = data_dir() / p
    if candidate.exists():
        return candidate
    return (source_repo() / p).resolve()
