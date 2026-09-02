# Development guide

## Module map

```
mcpsim/
  cli.py            click commands; clean-error wrapping; engine/offaxis flags
  config.py         dataclasses + YAML overlay (strict); presets; Config.tag/notes
  paths.py          source-repo resolution (MCPSIM_SOURCE_REPO / source_repo / sibling)
  model.py          fast-mode compute core -> ProductionResult; geometry-honest
                    acceptance dispatch (_meson_ageo)
  pipeline.py       run orchestration; regenerate_inputs (warning-returning);
                    write_tables (the golden-table format)
  geometry.py       theta cuts + the azimuthal kernels (face/circle_hit_fraction)
  physics/
    constants.py    meson specs, multiplicity tables, materials, DY constants
    meson_decay.py  I2/I3 + meson_yield (fidelity-aware)
    brem.py         cone-cut and off-axis grid integration
    drellyan.py     DY yield + row_mismatch
    sensitivity.py  D(eps), charge grid, exclusion_contour (the one solver)
    luminosity.py   effective luminosity
  generation/
    samples.py      archived PYTHIA parent loading (uproot)
    pydecay.py      the Python decay engine (samplers, kernel scan, cache)
    decay.py        cpp engine: lanl Dalitz + patched decayVectorMeson 2-body
    mesons.py       PYTHIA / Burman-Smith generation
    brem_run.py     VEGAS brem (patch-a-copy) + Condor bundles
    madgraph.py     DY generation + LHE/cross-section parsers
  validation/
    parity.py       Tests A/B/C vs the legacy C++  (python -m mcpsim.validation.parity)
    dy_audit.py     DY tiers 1-2                    (python -m mcpsim.validation.dy_audit)
  plotting/         production/limit plots + off-axis overlays + constraints
```

## The engines contract

Both decay engines produce the same artifact: a 2-column
`(mass_GeV, per-chi acceptance)` table consumed by
`model._load_acceptance_on_grid`. `pydecay.cached_acceptance` adds a JSON
sidecar (geometry, seed, fidelity, sample provenance) and caches by a hash of
everything the result depends on. Adding a third engine = producing that
table and wiring `decay.check_supported` + `model._meson_ageo`.

Wrapping read-only legacy code with hardcoded settings: regex-patch a copy
and fail loudly on zero substitutions — see `brem_run._prepare_patched_script`
and `decay._patched_vector_meson_binary`.

## Parity discipline

- `tests/golden/` are frozen outputs of the published pipeline, captured
  before this package refactored it — they *are* the baseline reference.
  `--fidelity legacy` at preset geometry must reproduce them **bit-for-bit**:
  `pytest -m golden`.
- Any deliberate physics change goes behind the fidelity switch and gets its
  delta quantified in the fidelity table in `docs/physics.md`.
- Engine changes must keep the parity harness within |z| ≤ 3
  (`pytest -m parity`, or `python -m mcpsim.validation.parity --quick`).
  These are development tools by design — they are deliberately NOT exposed
  on the user CLI.

## Tests

```
pytest                  # toolchain-free (CI): 60+ tests, mini-repo fixture
pytest -m golden        # needs the source repo
pytest -m parity        # needs ROOT + the PYTHIA samples (~10 min)
```

The `mini_repo` fixture (`tests/conftest.py`) is a 5-mass synthetic source
repo — extend it rather than pointing tests at the real data.

## Adding a preset

1. `mcpsim/presets/<name>.yaml` — comment to the standard of
   `lanl_12bar.yaml`/`flame.yaml` (provenance of every number).
2. If the geometry has no legacy acceptance files, set `channels`/`engine` so
   the Python engine computes them (see `flame.yaml`), and pick a `family`
   deliberately (it controls the Υ flat acceptance).
3. Add the preset name to `tests/test_cli.py::test_presets_lists_all`.
4. Run it once with `--fidelity legacy` and eyeball the warnings.

## Release checklist

- `pytest && pytest -m golden` green; `pytest -m parity` |z| ≤ 3.
- Any physics delta added to the fidelity table in `docs/physics.md`, with
  numbers.
- `mcpsim.__version__` bump (single source; pyproject reads it).
- Regenerate the data manifest if the data tree changed.
