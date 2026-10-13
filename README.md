# mcpsim

mcpsim computes **millicharged-particle (mCP) production and sensitivity** for
fixed-target and collider experiments. It runs from a config file: you set a
beam energy, a target, a detector geometry, and an exposure. Detectors can sit
off-axis. You get back production tables and plots of the baseline yield
N<sub>χ</sub>/ε², plus 95% CL exclusion contours in the (m<sub>χ</sub>, ε)
plane.

Four production channels are combined:

- meson 2-body decay (ρ, ω, φ, J/ψ, Υ → χχ̄)
- meson 3-body Dalitz decay (π⁰, η → γχχ̄)
- proton bremsstrahlung (precomputed eVMD (θ, p) grids)
- Drell-Yan (qq̄ → γ* → χχ̄)

The Υ is a special case. It is produced too rarely to have an acceptance scan
of its own, so it borrows the J/ψ's. Its contribution is around 10⁻⁴ of the
J/ψ. See [docs/physics.md](docs/physics.md).

Built-in presets: **DarkQuest/SpinQuest** (120 GeV), **SHiP** (400 GeV),
**LANSCE 12-bar demonstrator** (0.8 GeV), **FLAME @ FNAL** (120 GeV, 1 km,
off-axis).

## Install

**1. The code.**

```bash
git clone git@github.com:lbailloeul/mcpsim.git && cd mcpsim
pip install -e '.[dev]'          # numpy scipy matplotlib pyyaml click uproot
```

**2. The data.** The repository carries no data files. Download `data.zip`
(3.6 GB) from the data record on Zenodo,
[10.5281/zenodo.22690080](https://doi.org/10.5281/zenodo.22690080). It holds
everything mcpsim runs on: the precomputed acceptance scans, bremsstrahlung
grids, Drell-Yan files, published-constraint contours, and the PYTHIA parent
samples at both 120 and 400 GeV.

```bash
unzip data.zip -d ~/mcpsim-data
export MCPSIM_SOURCE_REPO=~/mcpsim-data     # add this to your shell profile
```

The archive already has the directory layout mcpsim expects, so there is
nothing to rearrange.

**3. Check it worked.**

```bash
mcpsim doctor                    # data location + toolchain status
mcpsim doctor --check-data       # also verifies sha256 for every data file
```

`--check-data` should report `354 ok, 0 missing, 0 corrupt`. The doctor will
list missing optional toolchains (ROOT, PYTHIA, MPI, MadGraph) — none of those
are needed for fast mode, which is what the Quick start below uses. See
[docs/install.md](docs/install.md) for the dependency tiers and troubleshooting.

## Quick start

```bash
mcpsim presets                                  # list built-in experiments
mcpsim run --preset darkquest                   # production + limit (fast mode)
mcpsim run --preset flame --offaxis 4,6.5,9     # off-axis scan + overlay plots
mcpsim run --preset darkquest --target molybdenum  # same setup, SHiP's target material
mcpsim compare -p darkquest -p ship             # both limits on one figure (the paper's plot)
mcpsim run --config my_experiment.yaml          # custom beam/target/detector
```

Outputs (PDF plots, numeric `.txt` tables, and a `run_<tag>.yaml` record of the
fully resolved configuration) land in `mcpsim-output/<experiment-tag>/`.
Physics caveats that apply to a run (borrowed acceptances, approximate
channels, validity limits) are printed as `!` lines at the end; the full list
is in [docs/physics.md](docs/physics.md#known-physics-caveats).

## The two decay engines

Meson-decay acceptance can be computed by two engines (`engine.decay` in the
config, `--engine` on the CLI):

- **python** (default) — vectorized re-decay of archived PYTHIA parent
  samples; handles any geometry including off-axis, needs no ROOT, runs in
  minutes.
- **cpp** — the legacy C++ decay binaries (Insung's `decayVectorMeson.cc`,
  `lanl_decayPion_12bar.cc`), compiled on demand; kept as the cross-check and
  for generating fresh samples at new beam energies.

Both are kept because comparing them caught real physics discrepancies that
neither would have revealed alone. Two came out of that comparison.

The first was the ρ mass. The Python 2-body decay initially used the fixed PDG
value, where the C++ uses the per-event Breit-Wigner mass. That was a 7%
acceptance shift, significant at 3–4σ, until the two were matched.

The second was the C++ Dalitz sampler, which carries angular-sampling quirks
worth up to ±8% in acceptance. The corrected sampler removes them.

With both understood, the engines agree within 1σ on identical parent samples.
That comparison is kept as a development check (`pytest -m parity`) rather than
something you need to run yourself.

Physics fidelity (`--fidelity`):

- **corrected** (default) — physically-correct samplers and constants.
- **legacy** — reproduces the published pipeline (the 3.14-vs-π Dalitz
  constant, the legacy sampler quirks, the cone-cut brem acceptance, the
  paper's iron Drell-Yan scans and 13 mb normalisation for every target, and
  the paper's r = 0.5 m disc for DarkQuest and SHiP, which otherwise model a
  1×1 m face).
  `pytest -m golden` enforces this against frozen tables.

The corrected−legacy differences are modest on the limits:
- **Dalitz:** +0.05% on yields, ≤±8% on acceptance, ≲1% on ε.
- **1×1 m face:** meson acceptance +17–33%, ε 1–3%.
- **Drell-Yan** (per-material tables, normalisation and the face): DY yield
  ×1.2–1.6, ε up to 10–13% where Drell-Yan dominates.

Each one is quantified in the fidelity table in
[docs/physics.md](docs/physics.md).

## Compute modes

- **Fast (default):** evaluates the physics on precomputed inputs. The
  acceptance always matches the geometry you asked for. Preset geometries load
  their precomputed files verbatim, and a custom geometry is recomputed by the
  Python engine. If the archived samples needed for that are missing, mcpsim
  says so plainly rather than quietly falling back on the preset's numbers.
- **`--regenerate`:** runs the heavy backends — PYTHIA or Burman-Smith meson
  generation, the VEGAS bremsstrahlung integrator (`mpirun`), the C++ decay
  step, MadGraph for Drell-Yan. Requires the `gen` toolchains.
- **`--condor`:** emit/submit HTCondor bundles for brem and Drell-Yan (one
  job per mass; LXPLUS-ready, with a `README_condor.txt` in each bundle).
  Exits with instructions if `condor_submit` is absent.

## Documentation

| Doc | Contents |
|---|---|
| [docs/install.md](docs/install.md) | dependency tiers, data bundle, source-repo resolution, troubleshooting |
| [docs/usage.md](docs/usage.md) | commands, output layout, the **full config-schema reference** |
| [docs/physics.md](docs/physics.md) | conventions: acceptance definition, detection model, thresholds, per-channel formulas, fidelity deltas |
| [docs/data.md](docs/data.md) | data-file manifest, formats, provenance status, known inconsistencies |
| [docs/development.md](docs/development.md) | module map, engines contract, tests, adding a preset |

## Tests

```bash
pytest                  # toolchain-free suite, no source repo needed
pytest -m golden        # bit-for-bit legacy parity (needs the source repo)
pytest -m parity        # Python engine vs C++ binaries (needs ROOT + samples)
```

## Additional resources

- **The predecessor repo** —
  [exoticdarksectors/decaysimulation](https://github.com/exoticdarksectors/decaysimulation):
  the earlier meson-decay → mCP simulation that mcpsim consolidates and
  refines. If you are new to the subject, start there — it walks through the
  PYTHIA setup step by step and carries the legacy 2-body and 3-body (Dalitz)
  decay scripts in their original standalone form, which the engines here
  reimplement and validate against.
- **PYTHIA 8.3** — [arXiv:2203.11601](https://arxiv.org/abs/2203.11601):
  the event generator behind the archived meson parent samples that both
  decay engines consume.
- **The physics** — [arXiv:2512.11027](https://arxiv.org/abs/2512.11027),
  *Dedicated Searches for Millicharged Particles at Intensity-Frontier
  Facilities: SpinQuest and SHiP*: the production channels, detection model,
  and sensitivity treatment this pipeline implements (the mcpsim-side
  conventions are spelled out in [docs/physics.md](docs/physics.md)).
