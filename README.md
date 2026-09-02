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
- Drell-Yan (qq̄ → γ* → χχ̄, precomputed MadGraph scans)

The Υ is a special case. It is produced too rarely to have an acceptance scan
of its own, so it borrows the J/ψ's. Its contribution is around 10⁻⁴ of the
J/ψ. See [docs/physics.md](docs/physics.md).

Built-in presets: **DarkQuest/SpinQuest** (120 GeV), **SHiP** (400 GeV),
**LANSCE 12-bar demonstrator** (0.8 GeV), **FLAME @ FNAL** (120 GeV, 1 km,
off-axis).

## Install

```bash
git clone <this-repo> mcpsim && cd mcpsim
pip install -e '.[dev]'          # numpy scipy matplotlib pyyaml click uproot
mcpsim install --dry-run         # "doctor": reports data + toolchain status
```

mcpsim also needs the **data source repo**: the precomputed acceptance, brem,
and DY inputs, about 355 MB for fast mode. Point `MCPSIM_SOURCE_REPO` at your
`github-repo-scripts` checkout or data bundle.

The DarkQuest and SHiP core inputs are publicly archived on the paper's Zenodo
record, [10.5281/zenodo.18330380](https://zenodo.org/records/18330380). That
covers the brem grids, the DY files, the acceptance scans, the legacy C++, and
the original UFO model. For the directory layout, the full bundle, and the
optional heavy toolchains (ROOT, PYTHIA, MPI, MadGraph), see
[docs/install.md](docs/install.md).

## Quick start

```bash
mcpsim presets                                  # list built-in experiments
mcpsim run --preset darkquest                   # production + limit (fast mode)
mcpsim run --preset flame --offaxis 4,6.5,9     # off-axis scan + overlay plots
mcpsim run --config my_experiment.yaml          # custom beam/target/detector
```

Outputs (PDF plots + numeric `.txt` tables) land in
`mcpsim-output/<experiment-tag>/`.

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
- **legacy** — reproduces the published pipeline **bit for bit** (the 3.14-vs-π
  Dalitz constant, the legacy sampler quirks, the cone-cut brem acceptance).
  `pytest -m golden` enforces this against frozen tables.

The corrected−legacy differences are small: +0.05% on Dalitz yields, ≤±8% on
Dalitz acceptance, ≲1% on ε. Each one is quantified in the fidelity table in
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
- **`--condor`:** emit/submit an HTCondor bundle for the brem step (one job
  per mass; LXPLUS-ready). Exits with instructions if `condor_submit` is
  absent.

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
