# Physics conventions

The single most important page for interpreting mcpsim output. Everything here
is implemented in `mcpsim/physics/` and `mcpsim/generation/pydecay.py`.

## The baseline yield N_χ/ε²

Production plots and tables show the **baseline yield**: the number of mCPs
crossing the detector face at ε = 1, i.e. N_χ/ε² (all production rates scale
as ε²). The ε dependence is applied only at the sensitivity stage.

## Acceptance convention (the ×2)

A **per-χ hit probability**: `a_geo = hits / (2 · N_decays)`, both χ's
counted — identical to the legacy C++ (`eff = hitCount/totalCount` in
`lanl_decayPion_12bar.cc`, filtered/total in `decayVectorMeson.cc`). The
factor 2 χ/decay is applied *downstream* in `meson_decay.meson_yield`:

    N_channel = N_POT · a_geo · 2 · c_meson · BR · I_phase-space

- `c_meson`: per-POT meson multiplicities (`C_MESON_BY_BEAM`, tabulated for
  120/400/0.8 GeV — see the provenance note below).
- `BR`: γγ branching for Dalitz parents, e⁺e⁻ for the vector mesons.
- `I₂(x, y)` (2-body) and `α·I₃(x)` (Dalitz) phase-space factors,
  x = (m_χ/M)².

Two-body decays use the **per-event Breit-Wigner mother mass** from the parent
sample (sub-threshold parents dropped from numerator *and* denominator) —
this matches `decayVectorMeson.cc` and matters for the 150 MeV-wide ρ
(~5–7% in a_geo vs a fixed-pole treatment).

The Υ is the one meson with no acceptance scan behind it. It is produced far
too rarely for a parent sample to be built: at 120 GeV the Υ comes out some
1.6×10⁴ times less often than the J/ψ (`C_MESON_BY_BEAM`), and even the J/ψ
sample we have holds only 4132 events, reused 600 times over. An Υ sample of
comparable size is a PYTHIA campaign in its own right, and attempts to produce
one have not been practical.

In its place the Υ borrows the J/ψ's acceptance: the **first (low-mass) value of
the J/ψ acceptance computed for that same run**, held flat across the mass grid.
At a preset geometry that comes from the archived scan — 0.011357 for DarkQuest,
0.011338 for SHiP, and both scans are genuinely flat there, over their first 68
and 109 rows. At an off-axis or custom geometry it is whatever the Python engine
just computed for *your* detector, so the borrowed number tracks the geometry
rather than being pinned to the preset's.

For a channel worth about 10⁻⁴ of the J/ψ yield this is a comfortable
approximation, and it has never shifted a limit. Two things would be worth
checking if an Υ sample ever becomes available. The J/ψ acceptance is not flat
across the full range: the same scans rise to 0.019448 (DarkQuest) and 0.04961
(SHiP) by their last row. And they stop at m_χ = 1.525 GeV, essentially the
J/ψ's own kinematic edge (m_J/ψ/2 = 1.55 GeV), while Υ → χχ̄ stays open out to
m_χ = 4.73 GeV — so past roughly the first third of the Υ's mass range we are
carrying a borrowed number into territory no scan has covered.

`engine.decay: python` does not produce an Υ acceptance directly — the engine
re-decays archived parents and there are none for the Υ. What it does do is
compute the J/ψ acceptance for your geometry, which is then what the Υ borrows.

Two things gate the channel (`model._upsilon_ageo`):

- **`family`** decides whether the Υ runs at all (`constants.UPSILON_FAMILIES`,
  currently DarkQuest and SHiP). An unrecognised family has no established basis
  for the substitution and skips the channel. The FLAME preset relies on this —
  its 1 km bar array is far enough from those geometries that the borrow was
  judged unsound.
- **A J/ψ acceptance must exist for the run.** Without one there is nothing to
  borrow from and the channel is skipped rather than falling back on a number
  measured at some other detector.

`data.upsilon_ageo` overrides both, if you have a better value.

## Off-axis acceptance: the azimuthal kernel

Production at a symmetric pp/pA collision is azimuthally uniform, so a χ
crossing the detector plane at radius r = L·tanθ is uniform in φ. The
probability of landing on a face centred at transverse offset d is the exact
arc fraction of the circle of radius r inside the face
(`geometry.face_hit_fraction`, rectangular; `circle_hit_fraction`,
cylindrical). This scores *every* forward χ instead of binary-counting on a
~4×10⁻⁸ sr face — variance reduction ~2πd/(face width). Validated against
brute-force counting at every offset including d = 0 (`tests/test_pydecay.py`).

The same kernel re-weights the brem (θ, p) grids off-axis, with each coarse
0.045-dex θ bin smeared over its width (41 slices) because the face subtends
a radial band narrower than a bin.

## Detection model and the exclusion threshold

    N_signal(ε, m) = [N_χ/ε²](m) · ε² · P(ε),
    P(ε) = (1 − exp(−n_γ ε²))^a

- `n_γ` = photoelectrons at ε = 1 for a full bar traversal. Either set it
  directly, or set `sensitivity.scintillator` and `detector.bar_length_m` and
  it is derived by **linear scaling in bar length** from a material anchor
  (`constants.SCINTILLATORS`): plastic 2.5×10⁵ @ 1.5 m, CeBr 5.0×10⁶ @ 1.5 m —
  each anchored at the length it was measured for. The linear scaling is an
  energy-deposition argument, sound near the anchor length; it does not model
  light attenuation, so do not extrapolate far from it. `--ngamma` overrides
  either from the CLI.
- `a` = number of layers in coincidence.
- Exclusion boundary: smallest ε with N_signal ≥ `n_chi_threshold`.

### `n_chi_threshold`: where the background assumption lives

Nothing else in the pipeline models background, so this one number carries the
whole assumption. It is the number of *detected* signal events needed to
exclude a point — set it and you have declared how clean you expect the search
to be.

The default **3.09** is the 95% CL upper limit for **0 observed events and 0
expected background**, so every standard limit here is a zero-background
projection. It comes from the unified construction of
[Feldman & Cousins (1998), `physics/9711021`](https://arxiv.org/abs/physics/9711021).

For a search with real background b, read the upper endpoint μ₂ from that
paper's tables rather than inventing a number. Tables II–IX give intervals
[μ₁, μ₂] for the Poisson signal mean at 68.27%, 90%, 95% and 99% CL, two
tables per level — so **Tables VI and VII are the 95% pair**, and 3.09 is
their n₀ = 0, b = 0 entry. Table XII gives the *sensitivity*, the average
upper limit over an ensemble with the expected background and no true signal,
which is usually the right thing to quote for a projection.

Once b is large enough to be Gaussian the tables converge on the one-sided
result, μ₂ ≈ 1.64√b at 95% CL — so a little under 2√b is a serviceable
estimate when you only need the scale. The LANL preset's 45 events is the
proposal's own background estimate, not a zero-background number.

In the unsaturated regime (n_γ ε² ≪ 1), N_signal ∝ n_γ^a ε^(2a+2), so the
limit scales extremely weakly with flux: ε_lim ∝ flux^(−1/(2a+2)) — for
a = 3, flux^(−1/8). This is why FLAME's limits move only ~1–2% across
4–14 mrad while the flux changes ~10%.

The light yield enters just as weakly: ε_lim ∝ n_γ^(−a/(2a+2)), or n_γ^(−3/8)
at a = 3. Worth keeping in mind when a preset borrows an `n_gamma` measured for
a different bar — being wrong by a factor of two costs under 30% on ε.

## Proton bremsstrahlung

Precomputed (θ, p) grids per mass (`Brem_<E>GeV_<mass>.txt`; header documents
the luminosity convention: production within one interaction length of the
target). Three σ columns = Λ_p ∈ {1.0, 1.5, 2.0} GeV form-factor cutoffs →
the brem_low/central/high band; the **central Λ_p = 1.5** enters the total.

**The canonical grids include the eVMD ρ/ω resonances.** The archived
`off_resonance` set (in `archived/condor-brem-mcp-production/`) was a one-off
demonstration, not for production use. Consequence: **do not naively sum the
brem channel with the PYTHIA vector-meson channel** — resonant production is
counted in both. The default presets that include both channels follow the
published pipeline's convention; treat the resonance region (m ≈ 0.2–0.5 GeV)
accordingly.

## Drell-Yan

    N_DY = N_POT · σ_DY(m) / σ_inel · a_geo(m),  σ_inel = 13 mb

(the legacy `σ·1e-12 / 13e-3` expression, now written with honest units;
numerically identical, regression-tested). σ_DY comes from archived MadGraph
LO scans (nCTEQ15 nPDF); the generating cards were not preserved, but the
**original UFO model is recovered** — the paper's Zenodo record
([10.5281/zenodo.18330380](https://zenodo.org/records/18330380)) ships
`Minimal_MCP.zip`, the FeynRules SM+χ model behind the scans (χ pdg 31, mass
`Mchi` in MASS[31], millicharge `qX` in FRBlock[1]; couples to γ *and* Z,
though the Z is irrelevant at these masses). Point `MCPSIM_MCP_MODEL` at its
unzipped directory for exact reruns; the in-repo `models/mcp_ufo` is the
minimal QED-only recreation with the same χ pdg. See
the DY audit (`python -m mcpsim.validation.dy_audit`, a development tool)
and [data.md](data.md) for the open provenance items.
The DY a_geo is a **preset-geometry file**; unlike the meson channels it is
not recomputed for custom geometries.

## Fidelity: corrected vs legacy

| Knob | legacy | corrected (default) | size of the difference |
|---|---|---|---|
| Dalitz I₃ constant | 3.14 | π | +0.051% on every Dalitz yield |
| Dalitz sampler | θ_V flat in [0,π], χ built about ẑ, unrotated boost (the C++ quirks) | isotropic γ*, χ about the boost axis | ≤±8% a_geo at 4 mrad, ≲1% on ε |
| brem bar-array acceptance | equal-area cone cut | exact face kernel + sub-binning | geometry-dependent, small on-axis |
| 2-body BW mother mass | identical in both (matches the C++) | — | (5–7% ρ a_geo vs naive fixed-pole; informational) |

`--fidelity legacy` at preset geometry reproduces the frozen golden tables
bit-for-bit (`pytest -m golden`).

## Known provenance gaps (inherited)

- The 120 GeV PYTHIA samples were generated with `HardQCD:all = on,
  PhaseSpace:pTHatMin = 2` — a
  hard-QCD-biased sample, not min-bias; the same bias underlies the published
  DarkQuest acceptances.
- DY: see [data.md](data.md) (`_iron` tags on the molybdenum SHiP preset,
  nCTEQ15/nCTEQ naming, unrecorded lhaid).
