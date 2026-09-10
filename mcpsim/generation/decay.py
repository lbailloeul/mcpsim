"""Decay step: produce per-mass geometric acceptance from a meson sample.

Wraps the C++ decay binaries, scanning the mCP mass grid and accumulating a
(mass, acceptance) file in the same format model.py consumes in fast mode.

Reproducible paths:
  * Dalitz parents (pi0/eta) on a bar-array detector -> lanl_decayPion_12bar.cc,
    which accepts distance + bar layout as arguments.

Limited paths (the legacy C++ hardcodes geometry, so a custom detector cannot be
honoured without modifying the source):
  * 2-body vector mesons (decayVectorMeson.cc, 4 fixed args), and
  * cylindrical detectors for any channel (the bar binary uses a fixed 5 cm bar).
These raise a clear error pointing back to fast mode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import paths
from ..config import Config
from ..physics.constants import MESONS
from . import common

# Mother masses in GeV for the decay binary argument.
_MOTHER_MASS_GEV = {"pi0": 0.1349768, "eta": 0.547862}


def check_supported(cfg: Config, meson: str) -> None:
    """Preflight: raise ToolchainError if this meson cannot be regenerated
    with the configured engine.

    Called by the pipeline BEFORE meson generation so an unsupported
    combination fails in milliseconds instead of after a 10M-trial PYTHIA run.

    cpp engine coverage: Dalitz parents on a bar-array detector
    (lanl_decayPion_12bar.cc, geometry-parametrized) and 2-body parents on an
    on-axis cylindrical detector (decayVectorMeson.cc with its detector
    globals patched per geometry, brem_run.py-style). Everything else —
    including any off-axis placement — needs the Python engine.
    """
    spec = MESONS[meson]
    det = cfg.detector

    if cfg.engine.decay == "python":
        from . import samples
        if not samples.have_samples(cfg, meson):
            raise common.ToolchainError(
                f"no archived PYTHIA parent sample for '{meson}' under "
                f"{samples.samples_dir(cfg)}; install the sample bundle, set "
                f"engine.samples_dir, or use engine.decay = 'cpp' with a "
                f"PYTHIA toolchain."
            )
        return

    # cpp engine
    if det.offaxis_mrad != 0.0:
        raise common.ToolchainError(
            "the legacy C++ decay binaries are on-axis only; an off-axis "
            "detector needs engine.decay = 'python'."
        )
    if spec.channel == "dalitz" and det.type == "bar_array":
        return
    if spec.channel == "two_body" and det.type == "cylindrical":
        return
    raise common.ToolchainError(
        f"engine 'cpp' cannot regenerate '{meson}' ({spec.channel}) for a "
        f"{det.type} detector: lanl_decayPion_12bar.cc covers Dalitz + "
        f"bar_array, decayVectorMeson.cc covers 2-body + cylindrical. Use "
        f"engine.decay = 'python' for other combinations."
    )


def run_decay_scan(cfg: Config, meson: str, source: Path, mass_grid: np.ndarray,
                   work_dir: Path, *, seed: int = 12345,
                   sampler: str = "dalitz") -> Path:
    """Scan the mass grid with the legacy C++ binaries and write an
    accumulated (mass, acceptance) file. Dispatches per channel:
    Dalitz -> lanl_decayPion_12bar.cc, 2-body -> patched decayVectorMeson.cc.
    """
    check_supported(cfg, meson)
    if MESONS[meson].channel == "two_body":
        return _run_vector_meson_scan(cfg, meson, source, mass_grid, work_dir)
    return _run_dalitz_scan(cfg, meson, source, mass_grid, work_dir,
                            seed=seed, sampler=sampler)


def _run_dalitz_scan(cfg: Config, meson: str, source: Path, mass_grid: np.ndarray,
                     work_dir: Path, *, seed: int = 12345,
                     sampler: str = "dalitz") -> Path:
    """Dalitz acceptance scan via lanl_decayPion_12bar.cc (text source)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    source = _as_text_source(Path(source), work_dir, meson)

    src = paths.legacy_dir() / "lanl_decayPion_12bar.cc"
    binary = common.compile_cpp(src, paths.CACHE_DIR / "bin" / "lanl_decayPion_12bar")

    work_dir.mkdir(parents=True, exist_ok=True)
    det = cfg.detector
    mother = _MOTHER_MASS_GEV[meson]
    out_acc = work_dir / f"acceptance_{meson}.txt"
    tmp_eff = work_dir / f"__tmp_eff_{meson}.txt"

    # Only scan kinematically-allowed masses: the Dalitz decay requires
    # 2*m_chi < mother mass; above threshold the acceptance is zero and the
    # binary rejects the point.
    allowed = np.atleast_1d(mass_grid)[np.atleast_1d(mass_grid) < mother / 2.0]
    if allowed.size == 0:
        raise common.ToolchainError(
            f"no masses below the {meson} Dalitz threshold ({mother/2:.4g} GeV) "
            f"in the selected grid."
        )

    rows = []
    for mchi in allowed:
        root_out = work_dir / f"mcp_{meson}_{mchi:.6g}.root"
        common.run([
            binary, str(root_out), str(mchi), str(tmp_eff), str(source),
            str(det.distance_m), str(det.bar_columns), str(det.bar_rows),
            str(mother), str(seed), sampler, "0",
            str(det.bar_layers), det.face_mode,
        ])
        eff = np.loadtxt(tmp_eff)
        rows.append((float(mchi), float(np.atleast_1d(eff)[1])))

    tmp_eff.unlink(missing_ok=True)
    arr = np.array(sorted(rows))
    np.savetxt(out_acc, arr, fmt="%.8g")
    return out_acc


# -- 2-body via patched decayVectorMeson.cc ---------------------------------
_VECTOR_MESON_PATCHES = (
    (r"(?m)^double\s+detectorRadius\s*=\s*[^;]+;",
     "double detectorRadius = {radius_m};"),
    (r"(?m)^double\s+distanceToBox\s*=\s*[^;]+;",
     "double distanceToBox = {distance_m};"),
)


def _patched_vector_meson_binary(cfg: Config, work_dir: Path) -> Path:
    """Compile decayVectorMeson.cc with its hardcoded detector globals
    replaced by this config's geometry (patch-a-copy, the same pattern
    brem_run.py uses on mCP_brem.py; the source repo stays read-only)."""
    import re

    det = cfg.detector
    src = paths.legacy_dir() / "decayVectorMeson.cc"
    if not src.exists():
        raise common.ToolchainError(f"decayVectorMeson.cc not found at {src}.")
    text = src.read_text()
    for pattern, repl in _VECTOR_MESON_PATCHES:
        new_text, n = re.subn(
            pattern,
            repl.format(radius_m=det.radius_m, distance_m=det.distance_m),
            text,
        )
        if n == 0:
            raise common.ToolchainError(
                f"could not patch expected global in decayVectorMeson.cc "
                f"(pattern {pattern!r}); the source layout may have changed."
            )
        text = new_text
    work_dir.mkdir(parents=True, exist_ok=True)
    patched = work_dir / "decayVectorMeson_patched.cc"
    patched.write_text(text)
    binary = (paths.CACHE_DIR / "bin"
              / f"decayVectorMeson_d{det.distance_m:g}_r{det.radius_m:g}")
    return common.compile_cpp(patched, binary)


def _run_vector_meson_scan(cfg: Config, meson: str, source: Path,
                           mass_grid: np.ndarray, work_dir: Path) -> Path:
    """2-body acceptance scan via the geometry-patched decayVectorMeson.cc.

    `source` may be a single ROOT file or a directory of them (mesonGen
    output); hits/totals are counted from the output trees with uproot and
    summed across files, so multi-file sources combine exactly.
    """
    common.require_module("uproot", "uproot is a core dependency — pip install -e .")
    import uproot

    work_dir.mkdir(parents=True, exist_ok=True)
    binary = _patched_vector_meson_binary(cfg, work_dir)
    roots = _root_sources(Path(source))
    spec = MESONS[meson]

    allowed = np.atleast_1d(mass_grid)[np.atleast_1d(mass_grid) < spec.mass / 2.0]
    if allowed.size == 0:
        raise common.ToolchainError(
            f"no masses below the {meson} 2-body threshold "
            f"({spec.mass/2:.4g} GeV) in the selected grid."
        )

    out_acc = work_dir / f"acceptance_{meson}.txt"
    rows = []
    for mchi in allowed:
        hits = total = 0
        for k, rf in enumerate(roots):
            out_root = work_dir / f"__vm_{meson}_{mchi:.6g}_{k}.root"
            eff_txt = work_dir / f"__vm_eff_{meson}.txt"
            common.run([binary, str(rf), str(out_root), str(mchi), str(eff_txt)])
            with uproot.open(out_root) as f:
                hits += f["mcp-filtered"].num_entries
                total += f["mcp"].num_entries
            out_root.unlink(missing_ok=True)
            eff_txt.unlink(missing_ok=True)
        rows.append((float(mchi), hits / total if total else 0.0))
    arr = np.array(sorted(rows))
    np.savetxt(out_acc, arr, fmt="%.8g")
    return out_acc


def _root_sources(source: Path) -> list:
    """ROOT files behind a source path (file, or a mesonGen output dir)."""
    if source.is_file() and source.suffix == ".root":
        return [source]
    if source.is_dir():
        roots = sorted(source.glob("*.root"))
        if roots:
            return roots
    raise common.ToolchainError(
        f"no ROOT meson source found at {source} (expected a .root file or a "
        f"directory of mesonGen outputs)."
    )


def _as_text_source(source: Path, work_dir: Path, meson: str) -> Path:
    """The lanl Dalitz binary reads text lines 'p_MeV cosTheta phi'.

    Burman-Smith already produces that; a PYTHIA mesonGen output (a directory
    of ROOT files, or a single ROOT file) is converted here. This also fixes
    a latent legacy bug where the PYTHIA directory path was handed to the
    C++ binary as if it were a text file."""
    if source.is_file() and source.suffix != ".root":
        return source
    common.require_module("uproot", "uproot is a core dependency — pip install -e .")
    import uproot

    out = work_dir / f"{meson}_source.txt"
    chunks = []
    for rf in _root_sources(source):
        with uproot.open(rf) as f:
            t = f["mesons"]
            branches = set(t.keys())
            if {"magnitude", "theta", "phi"} <= branches:
                a = t.arrays(["magnitude", "theta", "phi"], library="np")
                p, th, ph = a["magnitude"], a["theta"], a["phi"]
            else:
                a = t.arrays(["px", "py", "pz"], library="np")
                p = np.sqrt(a["px"] ** 2 + a["py"] ** 2 + a["pz"] ** 2)
                th = np.arccos(np.clip(a["pz"] / np.where(p > 0, p, 1.0), -1, 1))
                ph = np.arctan2(a["py"], a["px"])
            chunks.append(np.column_stack([p * 1e3, np.cos(th), ph]))
    np.savetxt(out, np.vstack(chunks), fmt="%.8g")
    return out
