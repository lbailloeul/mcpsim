# Using mcpsim

## Commands

| Command | Does |
|---|---|
| `mcpsim presets` | list built-in experiments |
| `mcpsim run` | production **and** limit: tables + plots (+ `--offaxis a,b,c` scans with overlay figures) |
| `mcpsim production` / `mcpsim limit` | a single plot (+ its tables); same options as `run`, incl. `--offaxis` scans |
| `mcpsim compare -p A -p B` | the limits of several runs on one figure (presets with `-p`, configs with `-c`), each solid with all channels and dashed without proton brem — the paper's DarkQuest + SHiP figure; writes every run's tables and limit curve file next to it |
| `mcpsim generate mesons\|brem\|dy` | one heavy generation step; exit ≠ 0 if nothing was produced; prints the `data:` block to use its outputs |
| `mcpsim doctor [--check-data]` | what this machine can run: Python modules, data location, UFO model, toolchains; `--check-data` verifies every data file (sha256) |
| `mcpsim install` | pip-install the extras, then the doctor report |

Common options: `--preset NAME` **or** `--config file.yaml` (not both — a config
names its base with `preset:`); `--out DIR` (default `mcpsim-output/<tag>/`);
`--channels meson_decay,brem,drell_yan` (aliases `mesons`, `dy`; unknown names
are rejected); `--engine python|cpp`; `--fidelity corrected|legacy`;
`--seed N`; `--quick`; `--ngamma N`; `--offaxis a,b,c`; `--target NAME` to
swap the target material — `iron` (DarkQuest), `molybdenum` (SHiP), `tungsten`,
`carbon` (graphite), or the aliases `fe`, `mo`, `moly`, `tzm`, `w`, `c`,
`graphite`.
A swap renames the run (`darkquest_molybdenum/`) so it never overwrites the
preset's outputs; it moves brem and Drell-Yan only — Drell-Yan reads the
built-in cross-section table for the new material (meson rates are per POT).
`--fidelity legacy` renames the run the same way (`darkquest_legacy/`), so the
paper reproduction and the default never overwrite each other.

## Output layout

```
mcpsim-output/<experiment-tag>/
  production_<tag>.txt   mass + N_chi/eps^2 per channel + total
  limit_<tag>.txt        mass + eps at the exclusion threshold (NaN = never)
  limit_curve_<tag>.txt  the limit curve alone, for collaborators to replot:
                         the masses where the threshold is reached, with a
                         header naming the experiment, beam, target, detector,
                         channels, statistics and signal model. Loads as it is
                         with np.loadtxt, ROOT's TGraph or gnuplot
  production_<tag>.pdf   per-channel production plot (+ .png)
  limit_<tag>.pdf        exclusion plot over the existing-constraint envelope,
                         with and without proton brem (+ .png)
  run_<tag>.yaml         the fully resolved config (preset + config + CLI
                         flags), mcpsim version and data location; rerun it
                         exactly with `mcpsim run --config run_<tag>.yaml`
  generated/             (only with --regenerate) fresh backend outputs
mcpsim-output/cache/
  acceptance/<hash>/     Python-engine acceptance cache + JSON provenance
  bin/                   compiled legacy C++ binaries
```

`mcpsim-output/` sits in the project when run from a source checkout, else in
the current directory. Off-axis scans with several angles add
`production_overlay_<tag>` / `limit_overlay_<tag>` figures in the base preset's
directory, and per-angle subdirectories (`<tag>_4mrad/`, ...); a single angle
writes straight into `--out`.

## Config schema — full reference

A YAML config overlays a **base preset**: named explicitly with `preset:`,
else inferred from the beam energy (≤1.5 GeV → lanl_12bar, ≥300 → ship, else
darkquest). An inferred preset whose energy differs from yours is reported as
a warning (its precomputed brem/DY inputs were made at *its* energy).

```yaml
name: my_experiment        # display name; slug becomes the output directory
preset: darkquest          # optional explicit base preset (typos raise)
family: darkquest          # upsilon flat-acceptance + plot-color key
                           #   known: darkquest, ship; anything else skips upsilon
source_repo: /path/to/data # optional override of MCPSIM_SOURCE_REPO

beam:
  energy_gev: 120.0        # must match a tabulated multiplicity set within 2%
                           #   (120 / 400 / 0.8 GeV) unless data.c_meson is given
  frame: fixed_target      # fixed_target | collider (collider disables brem+DY)
  n_pot: 1.0e20            # protons on target (collider: use energy_b_gev too)
  energy_b_gev: 0.0        # second beam energy, collider frame only

target:                    # brem (luminosity + sigma_pA rescale of the grids)
  name: iron               #   and Drell-Yan (per-material table + per-nucleon
                           #   normalisation); per-POT meson rates are
                           #   material-independent. Known names: iron,
                           #   molybdenum, tungsten, carbon (aliases fe, mo,
                           #   moly, tzm, w, c, graphite), in any case. A section
                           #   without a name keeps the base preset's material
                           #   and changes only the numbers it gives. Unknown
                           #   names REQUIRE the full four parameters (and
                           #   have no DY table):
  # Z: 26
  # A: 55.845
  # density_g_cm3: 7.87
  # interaction_length_cm: 16.8

detector:
  type: cylindrical        # cylindrical | bar_array
  distance_m: 40.0         # target -> detector plane
  radius_m: 0.5            # cylindrical face radius
  bar_columns: 4           # bar_array: bars per row (face width)
  bar_rows: 4              #   bars per column (face height)
  bar_layers: 3            #   depth layers (coincidence; enters sensitivity.a)
  bar_size_m: 0.05         #   transverse bar cross-section
  bar_length_m: 1.5        #   bar length along the line of sight (light yield
                           #   only — geometric acceptance uses the front face)
  face_mode: rectangular   #   passed through to the legacy C++ decay binary
  offaxis_mrad: 0.0        # transverse off-axis angle of the face centre;
                           #   nonzero needs engine python
# The DarkQuest and SHiP presets model a 1 x 1 m face (bar_array, 20 x 20 bars
# of 5 cm): meson acceptance is computed for it by the Python engine (~2.5 min
# on the first run, then cached), Drell-Yan acceptance comes from the tables.

legacy_detector:           # the published analysis's detector, when it differs
  type: cylindrical        #   (DarkQuest/SHiP: the paper's r = 0.5 m disc). The
  distance_m: 40.0         #   archived acceptance files were made for it, and
  radius_m: 0.5            #   fidelity legacy models it unless the config sets
                           #   `detector:` itself. Same keys as `detector`.

engine:
  decay: python            # python | cpp (see README "engines")
  fidelity: corrected      # corrected | legacy (bit-for-bit published numbers)
  samples_dir: null        # override the archived PYTHIA sample location
  seed: 20260727           # RNG seed of every engine, the C++ generators
                           #   included (acceptance cache key includes it)
  quick: false             # capped statistics for smoke runs (not cached)

sensitivity:               # detection D(eps) = (1-exp(-n_gamma eps^2))^a eps^2
  n_gamma: 2.5e5           # photoelectrons at eps = 1 (GEANT4-derived), or instead
  # scintillator: plastic  # derive n_gamma from a material anchor scaled
                           #   linearly with detector.bar_length_m
                           #   (plastic: 2.5e5 @ 1.5 m; cebr: 5.0e6 @ 1.5 m;
                           #   table in physics/constants.SCINTILLATORS).
                           #   Setting BOTH n_gamma and scintillator raises.
  a: 3                     # layer-coincidence exponent
  n_chi_threshold: 3.09    # detected events at the limit; carries the background
                           # assumption. 3.09 = 95% CL, zero background
                           # (Feldman-Cousins); see docs/physics.md to pick it
                           # for a search with real background.
  charge_min: 1.0e-5       # eps scan grid (log-spaced; min < max validated)
  charge_max: 1.0
  n_charge: 500

channels: [meson_decay, brem, drell_yan]

data:                      # fast-mode inputs; bare names resolve inside the
                           #   source repo (data-backup/ first, then the root)
  mass_grid: mship_values.txt
  acceptance:              # meson -> (mass, per-chi acceptance) file. It
                           #   merges per meson onto the base preset's: null
                           #   drops one, an unknown meson raises. A file a
                           #   config names is used as it is, and a detector
                           #   moved with --offaxis recomputes it.
    pi0: total_efficiency_outputdecayPionDARK.txt
  brem_dir: DarkQuest-brem-backup
  brem_pattern: "Brem_120GeV_*.txt"
  brem_max_theta_mrad: 12.5  # grids valid up to this angle (they carry the
                             #   generating detector's acceptance); warns beyond
  dy_cross: dy_cross_darkquest_nCTEQ15_iron.txt   # the paper's DY scan: 1 column,
  dy_ageo: ageo_darkquest_nCTEQ15_iron.txt        #   row-paired with this 2-column
                                                  #   per-chi acceptance. By default
                                                  #   sigma comes from the built-in
                                                  #   table for the target instead
                                                  #   (fidelity legacy: this scan);
                                                  #   setting dy_cross yourself keeps
                                                  #   your scan.
  dy_material: iron        # material of the dy_cross scan's nuclear PDFs
  c_meson:                 # per-POT meson multiplicities — REQUIRED at a
    pi0: 4.7               #   beam energy without tabulated values
    eta: 0.53
  upsilon_ageo: null       # explicit flat upsilon acceptance (overrides family)
```

Strictness: unknown keys at the top level or in any section **raise**, with a
"did you mean" suggestion (typos do not silently vanish); unknown channel names
raise; unknown target names raise unless fully specified; unknown explicit
presets raise; a `detector.type` (in `legacy_detector` too) other than
`cylindrical` or `bar_array`, and a `beam.frame` other than `fixed_target` or
`collider`, raise with a suggestion. Overriding `detector.bar_length_m`
re-derives n_gamma from an inherited `sensitivity.scintillator` unless you also
give `n_gamma`.

## Recipes

```bash
# Same experiment, different target material (outputs in darkquest_molybdenum/):
mcpsim run --preset darkquest --target molybdenum

# FLAME off-axis scan (the FNAL study):
mcpsim run --preset flame --offaxis 4,6.5,9,11.5,14

# Reproduce published DarkQuest numbers exactly (the paper's r = 0.5 m disc):
mcpsim run --preset darkquest --fidelity legacy

# The paper's limit figure, DarkQuest + SHiP (add --fidelity legacy for the
# published curves):
mcpsim compare -p darkquest -p ship

# Custom 60 m detector — acceptance recomputed for the real geometry:
mcpsim run --config examples/custom_60m.yaml

# Smoke-test a new geometry in ~10 s:
mcpsim run --preset flame --quick

# Fresh acceptance via the legacy C++ (needs ROOT): Dalitz parents on a bar
# face, 2-body parents on a disc (e.g. with --fidelity legacy); other
# combinations fall back to their precomputed input, with a message:
mcpsim generate mesons --preset darkquest --engine cpp --mass-subset 3

# Brem grid on LXPLUS:
mcpsim generate brem --preset darkquest --condor --n-proc 8

# Drell-Yan scan on LXPLUS, one MadGraph job per mass (README_condor.txt in
# the bundle covers MadGraph, the PDF set and the workers' Python):
mcpsim generate dy --preset darkquest --condor
```
