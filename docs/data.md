# Data files: manifest, formats, provenance

This page explains the data mcpsim runs on: what each input file contains,
where it came from, and which files to be careful with.

The authoritative machine-readable list, with sha256 checksums, is
[`data/MANIFEST.yaml`](../data/MANIFEST.yaml). `mcpsim install --check-data`
verifies a copy of the data against it.

How mcpsim finds a data file (`paths.resolve_data`): an absolute path is used
as given. A bare filename is looked for first in `<source-repo>/data-backup/`,
and then directly in `<source-repo>/`.

## File formats

| Kind | Format |
|---|---|
| mass grids (`mship_values.txt` 199 pts, `mship_values_copy.txt` 257 pts) | 1 column, GeV, log-spaced from 1 MeV to ~7 GeV. The `_copy` name is historical; SHiP simply uses the longer grid. |
| acceptance scans (`total_efficiency_*.txt`) | 2 columns, `mass_GeV a_geo`, where a_geo is the per-χ hit probability. The LANL files are an 11-column variant with a text header. mcpsim reads both. |
| brem grids (`Brem_<E>GeV_<mass>.txt`, 154 masses < 1 GeV) | 5 columns: `log10θ log10(p/GeV) σ@Λp=1.0 σ@1.5 σ@2.0`, in pb/bin. The 100×100 grid is VEGAS-adapted, so it differs from mass to mass. Each file carries a `#` header stating its luminosity convention. |
| DY pair (`dy_cross_*.txt`, `ageo_*.txt`) | The cross-section file is a single column of σ in pb, with **no mass column**. It pairs with the 2-column ageo file by row position. The ageo is the per-χ hit probability. |
| DY tables (`mcpsim/data/drell_yan/dy_sigma_<material>_<E>GeV.txt`, shipped with the package) | `# key: value` provenance header, then 3 columns: `m_chi[GeV] sigma[pb] acc` (per-nucleon σ at ε = 1, and the LO per-χ acceptance of the detector named in the `acc_detector` header line — the preset's 1×1 m face). Mass-keyed, read by interpolation. Built by `python -m mcpsim.generation.dy_tables --pdfsets DIR` (needs `pip install parton` and the nCTEQ15 LHAPDF grids); `--material`, `--energy` and `--detector-from PRESET` (the acceptance detector, default the Drell-Yan preset's at that energy) cover other cases — a new material must first be added to `physics/constants.TARGET_MATERIALS`. |
| constraint contours (`experiment-contours-small/*.csv`) | 2-column `mass,epsilon` CSV. SENSEI's mass column is in MeV rather than GeV, which the code flags. 8 of the 15 files are used; see `plotting/limits.py:CONSTRAINTS`. |
| PYTHIA parent samples (`mesongen-backup/output-data/debug*.root`) | TTree `mesons` holding `px,py,pz,mass,phi,theta,e,magnitude`. Generated at 120 GeV fixed-target with HardQCD, pTHatMin = 2. Entries: π⁰ 46.8M, η 5.3M, ω 6.1M, ρ 613k, φ 217k, J/ψ 4132. |

## Provenance status

How much is known about where each input came from, and how far you can chase
it back:

All 354 files in the manifest are published as `data.zip` on the data record,
[10.5281/zenodo.22690080](https://doi.org/10.5281/zenodo.22690080). Where the
table below says "publicly archived", it means the file *also* appears in the
paper's companion record ([18330380](https://zenodo.org/records/18330380)),
which is the citable provenance for the published inputs.

The manifest tracks data only. The generator and decay sources live in the
package (`mcpsim/legacy/`) and are versioned with the code, not checksummed
here.

| Input | What we know |
|---|---|
| brem grids | Fully traceable, and publicly archived on [Zenodo](https://zenodo.org/records/18330380) as two zips of 154 grids each. The headers document themselves, and they were produced by 154 Condor jobs, 8-way MPI, nitn = 10, neval = 4000. |
| LANL efficiency files | Fully traceable, but source-repo only — they are not on the Zenodo record. The geometry is the one in `lanl_12bar.yaml` (3x2x2 CeBr bars at ER1 = 6 m) and the filenames record the run: 5M trials, legacy fidelity. |
| DarkQuest/SHiP acceptance scans | Publicly archived on the same Zenodo record, but we cannot say exactly how each one was made. The files carry no headers, and while the generating scripts survive (`run_efficiency_tests*.sh`), nothing records which script produced which file. The preset comments note which variant each preset uses. |
| `C_MESON_BY_BEAM` | **Traced.** Each value is (entries in that beam's parent sample) / (trials that generated it), and all twelve reproduce the published table at both energies. The trial counts are not stored in the samples — they live in the generator's run script (pi0 5e4, eta 2e5, rho 1e6, omega 1e6, phi 1e7, jpsi 1e8), so the two must be kept in step by hand. The Υ entries cannot be checked this way: that generation was never run. |
| Υ acceptance | Not a stored constant. The Υ has no scan of its own — it is produced too rarely for a parent sample — so it borrows the first value of the J/ψ acceptance computed for the run in hand. At preset geometry that is 0.011357 (DarkQuest, from `total_efficiency_output2body_decay-jsi.txt`) or 0.011338 (SHiP, the `-SHiP` variant). See docs/physics.md. |
| DY files | Publicly archived; cards not preserved, but **the settings are now reconstructed** (2026-10-12): an independent LO convolution of `q q~ > chi chi~ / z` (q = u d s c) reproduces both `dy_cross` files to 0.1% (DarkQuest, all masses) and 0.2–0.7% (SHiP, up to 8 GeV) with **nCTEQ15_56_26 (iron bound proton, LHAPDF 100600) for both beams**, μ_F = μ_R = M(χχ̄), **M(χχ̄) ≥ 2 GeV**, α = 1/127.9 (UFO aEWM1) and the exact fixed-target √s (15.06 / 27.43 GeV). The same settings are now the defaults of `generation/madgraph.py`. Scripts: `mcpsim-output/dy_reproduction/`. The **acceptance files** hold the per-χ hit probability of LO (zero-pT) pairs: an exact LO calculation with the same PDFs matches DarkQuest's r = 0.5 m disc at 40 m to 1–5%; SHiP's matches a 1 m² face at 100 m (a 1×1 m square, or an equal-area disc — LO cannot tell them apart) up to 7.04 GeV and the r = 0.5 m disc from 7.4 GeV on (the extension of the longer mass grid). The size-labelled variants (`ageo_dy_*_0.5m*`, `*_1m*`) fit 1×1 m and 2×2 m faces best, with at-least-one-χ counting, so their labels are half-widths. |
| DY tables | Generated by mcpsim (2026-10-12) from LHAPDF grids nCTEQ15_1_1, _12_6, _56_26, _84_42, _108_54, _184_74 (central members); each file's header records its PDFs, settings, generator version and date. Default fidelity reads σ and the acceptance from them; legacy uses the archived scans and acceptance files. |

## Open questions

- ~~**`_iron` DY files on the molybdenum SHiP preset.**~~ **Resolved
  (2026-10-12):** a deliberate approximation — the published Drell-Yan used
  iron nuclear PDFs for DarkQuest and SHiP alike (confirmed by the authors).
  Declared as `data.dy_material: iron`; by default mcpsim now uses its
  molybdenum table for SHiP (docs/physics.md), and legacy fidelity the
  archived scan.
- **The DY acceptance counts one χ per pair, and SHiP's mixes two detector
  faces.** Found 2026-10-12 by reproducing the files at LO (see the provenance
  table above). The meson channels count both χ (2 × per-χ hit probability),
  so DY is on a different footing by a factor 2 (1.3–2× against counting
  events with at least one χ). SHiP's file switches from a 1 m² face (1×1 m
  square or equal-area disc) to the preset's r = 0.5 m disc between 7.04 and
  7.40 GeV.
  Both are inherited from the archived files and left as they are; they need
  a decision on the intended convention and geometry. The geometry mix only
  affects `--fidelity legacy`: by default DarkQuest and SHiP model a 1×1 m face
  and take the Drell-Yan acceptance from the tables.
- ~~**nCTEQ15 vs nCTEQ tags within the SHiP DY pair; unrecorded `lhaid`.**~~
  **Resolved (2026-10-12):** the SHiP cross sections are reproduced with
  nCTEQ15_56_26 (lhaid 100600), so the tags name the same set; the old
  `lhaid = 30000` placeholder (not a valid LHAPDF id) is replaced by 100600.
- **The Υ borrows its acceptance from the J/ψ.** Where the number comes from is
  settled (see above), and for a channel worth ~10⁻⁴ of the J/ψ the
  substitution is a fair one. The Υ is produced some 1.6×10⁴ times less often,
  so building a parent sample to compute a real scan from has not been
  feasible. What is untested is how far the borrowed value travels. The J/ψ
  scan ends at m_χ = 1.525 GeV, while Υ decays stay open out to 4.73 GeV, so
  most of the Υ's range uses a value carried over flat. Worth revisiting if an
  Υ sample ever appears — the Python engine cannot help without one.

## Things to watch out for

- The full source tree's `data-backup/` holds **zero-byte files left by failed
  runs** (`TwoByTwoefficiency-*`, `efficiency_outputdecayPion-4by4.txt`, and
  others). Nothing in the name distinguishes them from real output. The
  manifest lists only the good ones.
- Several competing DY files sit alongside the ones actually in use
  (`dy_cross_091525.txt`, `ageo_dy_dark-1m.txt`, and more), with no index
  saying which is which. The presets name the canonical pair explicitly.
- The source tree is not under version control anywhere. The bundle and its
  checksummed manifest are the durable record, so regenerate them after any
  change to the data.
