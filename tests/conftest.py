"""Shared fixtures: a synthetic miniature source repo so the fast-mode
pipeline can be exercised without the real 11 GB data tree."""

import numpy as np
import pytest

MASSES = np.array([0.005, 0.02, 0.05, 0.2, 1.0])


@pytest.fixture
def mini_repo(tmp_path, monkeypatch):
    """A tiny github-repo-scripts stand-in; returns its root path.

    Contains: a 5-point mass grid, 2-column acceptance files for all six
    mesons, one synthetic Brem_120GeV file in the standard 5-column format,
    and a row-aligned DY pair. MCPSIM_SOURCE_REPO points here for the test.
    """
    repo = tmp_path / "mini-source-repo"
    data = repo / "data-backup"
    data.mkdir(parents=True)

    np.savetxt(data / "mship_values.txt", MASSES)

    for meson in ("pi0", "eta", "rho", "omega", "phi", "jpsi"):
        np.savetxt(data / f"acc_{meson}.txt",
                   np.column_stack([MASSES, np.full(MASSES.size, 1e-3)]))

    brem_dir = repo / "mini-brem"
    brem_dir.mkdir()
    # log10(theta), log10(p), sigma@Lambda_p=1.0/1.5/2.0 [pb/bin]
    log10t = np.repeat(np.linspace(-4, -0.5, 8), 3)
    log10p = np.tile(np.linspace(-1, 2, 3), 8)
    sig = np.full((24, 3), 0.5)
    np.savetxt(brem_dir / "Brem_120GeV_0.02.txt",
               np.column_stack([log10t, log10p, sig]))

    np.savetxt(data / "dy_cross.txt", np.full(MASSES.size, 2.0))     # pb
    np.savetxt(data / "dy_ageo.txt",
               np.column_stack([MASSES, np.full(MASSES.size, 0.01)]))

    monkeypatch.setenv("MCPSIM_SOURCE_REPO", str(repo))
    return repo


@pytest.fixture
def mini_config(mini_repo):
    """A Config wired to the mini repo (120 GeV, all channels)."""
    from mcpsim.config import Config, DataConfig

    return Config(
        name="mini",
        data=DataConfig(
            mass_grid="mship_values.txt",
            acceptance={m: f"acc_{m}.txt"
                        for m in ("pi0", "eta", "rho", "omega", "phi", "jpsi")},
            brem_dir="mini-brem",
            brem_pattern="Brem_120GeV_*.txt",
            dy_cross="dy_cross.txt",
            dy_ageo="dy_ageo.txt",
        ),
    )
