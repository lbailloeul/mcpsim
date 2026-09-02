# Using mcpsim

## Commands

| Command | Does |
|---|---|
| `mcpsim presets` | list built-in experiments |
| `mcpsim run` | production **and** limit: tables + plots (+ `--offaxis a,b,c` scans with overlay figures) |
| `mcpsim production` / `mcpsim limit` | a single plot (+ its tables) |
| `mcpsim generate mesons\|brem\|dy` | one heavy generation step; exit ≠ 0 if nothing was produced |
| `mcpsim install` | doctor: pip extras, data resolution, toolchain report |

Common options: `--preset NAME` or `--config file.yaml`; `--out DIR`
(default `mcpsim-output/<tag>/`); `--channels meson_decay,brem,drell_yan`;
`--engine python|cpp`; `--fidelity corrected|legacy`; `--seed N`; `--quick`.

## Output layout

```
mcpsim-output/<experiment-tag>/
  production_<tag>.txt   mass + N_chi/eps^2 per channel + total
  limit_<tag>.txt        mass + eps at the exclusion threshold (NaN = never)
  production_<tag>.pdf   per-channel production plot
  limit_<tag>.pdf        exclusion plot over the existing-constraint envelope
  generated/             (only with --regenerate) fresh backend outputs
mcpsim-output/cache/
  acceptance/<hash>/     Python-engine acceptance cache + JSON provenance
  bin/                   compiled legacy C++ binaries
```

Off-axis scans add `production_overlay_<tag>` / `limit_overlay_<tag>` figures
in the base preset's directory, and per-angle subdirectories
(`<tag>_4mrad/`, ...).

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

target:                    # brem luminosity only; per-POT meson rates are
  name: iron               #   material-independent. Known names: iron,
                           #   molybdenum, tungsten, carbon. Unknown names
                           #   REQUIRE the full four parameters:
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

engine:
  decay: python            # python | cpp (see README "engines")
  fidelity: corrected      # corrected | legacy (bit-for-bit published numbers)
  samples_dir: null        # override the archived PYTHIA sample location
  seed: 20260727           # engine RNG seed (acceptance cache key includes it)
  quick: false             # capped statistics for smoke runs (not cached)

sensitivity:               # detection D(eps) = (1-exp(-n_gamma eps^2))^a eps^2
  n_gamma: 2.5e5           # photoelectrons at eps = 1 (GEANT4-derived) — OR:
  scintillator: plastic    # derive n_gamma from a material anchor scaled
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
  acceptance:              # meson -> (mass, per-chi acceptance) file
    pi0: total_efficiency_outputdecayPionDARK.txt
  brem_dir: DarkQuest-brem-backup
  brem_pattern: "Brem_120GeV_*.txt"
  dy_cross: dy_cross_darkquest_nCTEQ15_iron.txt   # 1 column, row-paired
  dy_ageo: ageo_darkquest_nCTEQ15_iron.txt        #   with this 2-column file
  c_meson:                 # per-POT meson multiplicities — REQUIRED at a
    pi0: 4.7               #   beam energy without tabulated values
    eta: 0.53
  upsilon_ageo: null       # explicit flat upsilon acceptance (overrides family)
```

Strictness: unknown keys in any section **raise** (typos do not silently
vanish); unknown target names raise unless fully specified; unknown explicit
presets raise.

## Recipes

```bash
# FLAME off-axis scan (the FNAL study):
mcpsim run --preset flame --offaxis 4,6.5,9,11.5,14

# Reproduce published DarkQuest numbers exactly:
mcpsim run --preset darkquest --fidelity legacy

# Custom 60 m detector — acceptance recomputed for the real geometry:
mcpsim run --config examples/custom_60m.yaml

# Smoke-test a new geometry in ~10 s:
mcpsim run --preset flame --quick

# Fresh acceptance via the legacy C++ (2-body, cylindrical, needs ROOT):
mcpsim generate mesons --preset darkquest --engine cpp --mass-subset 3

# Brem grid on LXPLUS:
mcpsim generate brem --preset darkquest --condor --n-proc 8
```
