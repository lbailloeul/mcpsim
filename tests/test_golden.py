"""Legacy-parity golden tests (opt-in: pytest -m golden; needs the source repo).

fidelity='legacy' with untouched preset geometry must reproduce
tests/golden/ bit-for-bit — the frozen output of the published pipeline as it
stood before this package refactored it. Any diff here means a
physics-affecting change leaked into the legacy path.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mcpsim import paths
from mcpsim.config import EngineConfig, load_preset
from mcpsim.model import compute_production
from mcpsim.pipeline import write_tables

GOLDEN = Path(__file__).parent / "golden"

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(not paths.source_repo().exists(),
                       reason="source repo not available"),
]


@pytest.mark.parametrize("preset", ["darkquest", "ship", "lanl_12bar"])
def test_legacy_fidelity_reproduces_golden(preset, tmp_path):
    cfg = load_preset(preset)
    cfg = replace(cfg, engine=replace(cfg.engine, fidelity="legacy"))
    result = compute_production(cfg)
    write_tables(cfg, result, tmp_path)
    for kind in ("production", "limit"):
        new = np.loadtxt(tmp_path / f"{kind}_{cfg.tag}.txt")
        gold = np.loadtxt(GOLDEN / cfg.tag / f"{kind}_{cfg.tag}.txt")
        np.testing.assert_array_equal(new, gold, err_msg=f"{preset} {kind}")
