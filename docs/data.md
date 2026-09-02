# Data files: manifest, formats, provenance

This page explains the data mcpsim runs on: what each input file contains,
where it came from, and which files to be careful with.

The authoritative machine-readable list, with sha256 checksums, is
[`data/MANIFEST.yaml`](../data/MANIFEST.yaml). Bundles are built and verified
against it with `tools/make_data_bundle.py` and `mcpsim install --check-data`.

How mcpsim finds a data file (`paths.resolve_data`): an absolute path is used
as given. A bare filename is looked for first in `<source-repo>/data-backup/`,
and then directly in `<source-repo>/`.

## File formats

| Kind | Format |
|---|---|
| mass grids (`mship_values.txt` 199 pts, `mship_values_copy.txt` 257 pts) | 1 column, GeV, log-spaced from 1 MeV to ~7 GeV. The `_copy` name is historical; SHiP simply uses the longer grid. |
| acceptance scans (`total_efficiency_*.txt`) | 2 columns, `mass_GeV a_geo`, where a_geo is the per-χ hit probability. The LANL files are an 11-column variant with a text header. mcpsim reads both. |
| brem grids (`Brem_<E>GeV_<mass>.txt`, 154 masses < 1 GeV) | 5 columns: `log10θ log10(p/GeV) σ@Λp=1.0 σ@1.5 σ@2.0`, in pb/bin. The 100×100 grid is VEGAS-adapted, so it differs from mass to mass. Each file carries a `#` header stating its luminosity convention. |
| DY pair (`dy_cross_*.txt`, `ageo_*.txt`) | The cross-section file is a single column of σ in pb, with **no mass column**. It pairs with the 2-column ageo file by row position. |
| constraint contours (`experiment-contours-small/*.csv`) | 2-column `mass,epsilon` CSV. SENSEI's mass column is in MeV rather than GeV, which the code flags. 8 of the 15 files are used; see `plotting/limits.py:CONSTRAINTS`. |
| PYTHIA parent samples (`mesongen-backup/output-data/debug*.root`) | TTree `mesons` holding `px,py,pz,mass,phi,theta,e,magnitude`. Generated at 120 GeV fixed-target with HardQCD, pTHatMin = 2. Entries: π⁰ 46.8M, η 5.3M, ω 6.1M, ρ 613k, φ 217k, J/ψ 4132. |

## Provenance status

How much is known about where each input came from, and how far you can chase
it back:

Before reading the table, one thing to keep straight: "publicly archived" means
the file itself is on the Zenodo record, which covers 328 of the 353 files in
the manifest. The mass grids, the constraint contours, the LANL pipeline, the
PYTHIA parent samples, and a few small config and notes files are in the source
repo only — see [install.md](install.md) for the full list. Several of the
files cited below as evidence of provenance fall in that group.

| Input | What we know |
|---|---|
| brem grids | Fully traceable, and publicly archived on [Zenodo](https://zenodo.org/records/18330380) as two zips of 154 grids each. The headers document themselves, and `data-backup/LXPLUS_BREM.md` describes how they were produced: 154 Condor jobs, 8-way MPI, nitn = 10, neval = 4000. That write-up is not on the record, so a Zenodo download gives you the grids without it. |
| LANL efficiency files | Fully traceable, but source-repo only — they are not on the Zenodo record. `lanl_12bar_pipeline/README.md` records the geometry and the number of trials. |
| DarkQuest/SHiP acceptance scans | Publicly archived on the same Zenodo record, but we cannot say exactly how each one was made. The files carry no headers, and while the generating scripts survive (`run_efficiency_tests*.sh`), nothing records which script produced which file. The preset comments note which variant each preset uses. |
| `C_MESON_BY_BEAM` | No provenance at all. These are recorded only as "from PYTHIA", with no run to reproduce them from, so they stand as defining constants of the pipeline. |
| `UPSILON_AGEO_DEFAULT` | Known exactly. Both values come from the low-mass end of the family's own J/ψ scan: 0.011357 from `total_efficiency_output2body_decay-jsi.txt`, and 0.011338 from the `-SHiP` variant. The Υ is produced too rarely to have a scan of its own, so it borrows the J/ψ's. See docs/physics.md. |
| DY files | Publicly archived, but only partly traceable. The MadGraph cards were not preserved. The original UFO model (`Minimal_MCP.zip`) is on the record, though, so a rerun can now settle the questions below. The PDF set is known only from the filenames. |

## Open questions

- **`_iron` DY files on the molybdenum SHiP preset.** Either an iron-nPDF
  approximation or a mislabel. The data alone cannot tell us which. Rerunning
  with the recovered `Minimal_MCP` model would settle it (docs/physics.md,
  tier 3).
- **nCTEQ15 vs nCTEQ tags within the SHiP DY pair.** We assume the two tags
  mean the same set. The `lhaid` was never recorded, and the default of 30000
  in `madgraph.py` is a placeholder.
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
