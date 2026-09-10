# Installing mcpsim

## Dependency tiers

| Tier | What you can do | Requirements |
|---|---|---|
| **1. Fast mode** | all presets, custom configs at preset geometries, plots/tables/limits | `pip install -e .` + the **fast-mode data bundle** |
| **2. Python engine** | acceptance for *any* geometry incl. off-axis (FLAME scans, custom detectors) | tier 1 + the **PYTHIA sample bundle** (~3 GB) |
| **3. cpp engine / regeneration** | fresh meson samples at new beam energies, C++ cross-checks, VEGAS brem, Condor bundles | tier 2 + ROOT (+ PYTHIA 8, MPI) |
| **4. Drell-Yan generation** | new DY scans | tier 3 + MadGraph5_aMC@NLO + LHAPDF + the mCP UFO model (`models/mcp_ufo`) |

Python ≥ 3.9. Core pip deps (installed by `pip install -e .`): numpy, scipy,
matplotlib, pyyaml, click, **uproot** (reads the ROOT sample files — no ROOT
installation needed for tiers 1–2).

## The data source repo

mcpsim reads precomputed inputs from a separate directory tree (historically
`github-repo-scripts`). It is located by, in order:

1. the `MCPSIM_SOURCE_REPO` environment variable,
2. a `source_repo:` key in your YAML config,
3. a `github-repo-scripts` sibling of the mcpsim checkout.

The tree only needs to contain what your runs use (see
[data.md](data.md) for the manifest):

- `data-backup/` — mass grids, acceptance scans, DY files (~424 KB)
- `DarkQuest-brem-backup/`, `SHiP-brem-backup/` — brem (θ,p) grids (~354 MB)
- `experiment-contours-small/` — published-constraint contours (~332 KB)
- `mesongen-backup/output-data/debug*.root` — PYTHIA parent samples (~3 GB,
  tier 2 only)
- `lanl_12bar_pipeline/output/` — LANL efficiency scans (~24 KB)

The tree holds **data only**. Every generator and decay source mcpsim runs is
shipped inside the package (`mcpsim/legacy/`), so tiers 3-4 need the external
toolchains but no extra sources.

Verify a machine's copy with `mcpsim install --check-data`, which checks
sha256 for every file against `data/MANIFEST.yaml`.

### Public data records (Zenodo)

**The data record — [10.5281/zenodo.22690080](https://doi.org/10.5281/zenodo.22690080)**
(*data files for mcpsim CLI*, CC-BY 4.0) is what you want. It is a single
`data.zip` (3.6 GB) holding all 354 files in `data/MANIFEST.yaml`, already in
the layout mcpsim expects:

```bash
unzip data.zip -d ~/mcpsim-data
export MCPSIM_SOURCE_REPO=~/mcpsim-data
mcpsim install --check-data          # 354 ok, 0 missing, 0 corrupt
```

Nothing needs rearranging, and nothing is missing — the mass grids, contour
CSVs, LANL scans and both 120 and 400 GeV PYTHIA sample sets are all in it.

**The paper's companion record — [10.5281/zenodo.18330380](https://zenodo.org/records/18330380)**
(*Data and Code for "Dedicated Searches for Millicharged Particles at
Intensity-Frontier Facilities: SpinQuest and SHiP"*) archives the inputs behind
the publication: the two brem grid sets, the DY files, the DarkQuest/SHiP
acceptance scans, the legacy C++, and **`Minimal_MCP.zip`, the original
FeynRules UFO model** the archived DY scans were generated with.

You do not need it to run mcpsim — everything there that mcpsim uses is in
`data.zip`, and the legacy sources ship inside the package (`mcpsim/legacy/`).
The one thing it has that `data.zip` does not is `Minimal_MCP.zip`, needed only
if you regenerate Drell-Yan against the original model (see docs/physics.md).

## The doctor

```bash
mcpsim install --dry-run
```

reports: importable Python modules, how the source repo resolved (and whether
it exists), the `MCPSIM_MCP_MODEL` UFO setting, and which external toolchains
(`root-config`, `pythia8-config`, `mpirun`, `mg5_aMC`) are on PATH — including
a scan of your conda envs for ones that carry them.

## External toolchains (tier 3+)

- **ROOT** (compiles the legacy C++): `conda install -c conda-forge root` or
  `brew install root`; on LXPLUS source an LCG view
  (`/cvmfs/sft.cern.ch/lcg/views/LCG_106/*/setup.sh`).
- **PYTHIA 8** (fresh meson samples): `conda install -c conda-forge pythia8`
  provides `pythia8-config`.
- **MPI + vegas** (local brem): `pip install -e '.[gen]'` (mpi4py, vegas) plus
  an MPI runtime (`brew install open-mpi` / `conda install -c conda-forge openmpi`).
  On LXPLUS prefer `--condor` (no local MPI needed).
- **MadGraph + LHAPDF** (DY): install MG5_aMC, put `mg5_aMC` on PATH, install
  LHAPDF with the nCTEQ15 grids, and set `MCPSIM_MCP_MODEL` to the UFO model
  directory (this repo ships one under `models/mcp_ufo`).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Source repo not found at ...` | data tree missing/mislocated | set `MCPSIM_SOURCE_REPO` to your bundle/checkout |
| `mass grid ... not found` | partial data tree | `mcpsim install --check-data` to see what's missing |
| `no brem files matched ...` warning | brem grids absent for this energy | install the brem set, or `--channels meson_decay` |
| `acceptance uses the PRESET-geometry file` warning | custom geometry but no PYTHIA samples | install the sample bundle (tier 2) or accept the approximation |
| `archived ... samples were generated at 120 GeV` | non-120 GeV beam with the python engine | generate fresh samples (tier 3) and set `engine.samples_dir` |
| `'root-config' not found` | ROOT not sourced | activate the conda env / LCG view that has it (`mcpsim install` lists candidates) |
| `Generation step '...' produced no outputs` (exit 1) | missing toolchain — the reasons are in the yellow warnings above it | install the tool or use the python engine |
| empty plots + `no channel produced any yield` | all channels skipped (check warnings) | fix the data paths listed in the warnings |
