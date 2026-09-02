"""Decay-engine parity vs the legacy C++ (opt-in: pytest -m parity).

Needs ROOT (compiles the legacy binaries) and the archived PYTHIA samples.
Thin wrapper over mcpsim.validation.parity (also runnable directly:
python -m mcpsim.validation.parity).
"""

import shutil

import pytest

from mcpsim import paths

pytestmark = [
    pytest.mark.parity,
    pytest.mark.skipif(not paths.source_repo().exists(),
                       reason="source repo not available"),
    pytest.mark.skipif(shutil.which("root-config") is None,
                       reason="ROOT not available"),
]


def test_engine_parity_vs_cpp():
    from mcpsim.validation import parity

    rows = parity.run_all(quick=False)
    checked = [r for r in rows if r.test.startswith(("A", "B"))]
    assert checked, "no parity comparisons ran"
    for r in checked:
        assert abs(r.z) <= 3.0, (
            f"{r.test} m={r.m_chi}: cpp={r.eff_cpp:.4e} py={r.eff_py:.4e} "
            f"z={r.z:+.1f}"
        )
