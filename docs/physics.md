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

**The target enters twice.** The effective luminosity counts nuclei in one
interaction length (ρλ/A, i.e. N_POT/σ_inel), and every grid entry already
contains σ_pA(A) = 43.55 mb A^0.7111 − 3.84 mb A^0.35 for the nucleus the grid
was generated for (its header line `A_<X> = … g/mol`). The readers rescale each
grid by σ_pA(A_target)/σ_pA(A_grid) (`physics.brem.target_rescale`), and the
model reports the factor. Done consistently the two nearly cancel — a thick
target gives about one interaction per POT — so iron and graphite differ by
~13% in brem. Changing only the luminosity is what went wrong before: graphite
luminosity on the iron grids overstated brem ×3.0. Grids whose header does not
declare A are read unscaled, with a warning.

**The grids also carry a detector.** They store σ × the DarkQuest
`accept_event` weight (single track or shared bar on a 1×1 m face at 40 m), not
the inclusive χ distribution. Off-axis beyond 12.5 mrad the weight is ≈ 0 at low
mass, and below it both-χ hits are undercounted; do not reuse them for another
geometry without regenerating. `data.brem_max_theta_mrad` (12.5 for the
DarkQuest grids, 5 for SHiP's) makes a run warn when any part of the face
reaches beyond that angle.

**The canonical grids include the eVMD ρ/ω resonances.** The archived
`off_resonance` set (in `archived/condor-brem-mcp-production/`) was a one-off
demonstration, not for production use. Consequence: **do not naively sum the
brem channel with the PYTHIA vector-meson channel** — resonant production is
counted in both. The default presets that include both channels follow the
published pipeline's convention; treat the resonance region (m ≈ 0.2–0.5 GeV)
accordingly.

## Drell-Yan

    N_DY = N_POT · σ_DY(m) / σ_inel · a_geo(m),   σ_inel = σ_inel(p-target)/A = 1/(ρ λ N_A)

σ_DY is a per-nucleon cross section, so it is divided by the per-nucleon share
of the **configured target's** interaction cross section — equivalently,
multiplied by the nucleons/cm² in one interaction length (paper eq. 4.14):
12.6 mb for iron, 10.7 for molybdenum, 8.6 for tungsten, 19.4 for graphite.
The published pipeline used 13 mb (the iron value, written `σ·1e-12 / 13e-3`)
for every target; `--fidelity legacy` keeps it.

**Per-material cross sections (default).** σ_DY comes from built-in tables
for the configured target and beam energy — iron, molybdenum, tungsten and
carbon at 120 and 400 GeV (`mcpsim/data/drell_yan/`), so SHiP's molybdenum
preset and `--target` change it. They are mcpsim's LO convolution
(`physics/dy_lo.py`, built by `python -m mcpsim.generation.dy_tables`) with
the archived scans' settings (below) except for the parton densities:

- beam: `nCTEQ15_1_1`, the free proton of the nCTEQ15 fit;
- target: the per-nucleon nucleus PDF (Z f^{p/A} + (A−Z) f^{n/A})/A, i.e.
  `nCTEQ15FullNuc_A_Z` (reproduced to 5×10⁻⁹), with the bound neutron from
  isospin. Molybdenum (A = 96) has no fitted set: the bound-proton PDF, which
  depends on A only, is interpolated in ln A between A = 84 and 108 (< 0.2%
  error, tested by predicting A = 84 from 56 and 108).

Each table also carries the acceptance: the LO per-χ acceptance of the
preset's detector, the 1×1 m face at 40 m (120 GeV) or 100 m (400 GeV),
computed with the same PDFs, which set the pair's rapidity
(`dy_lo.acceptance_per_chi`: pair at zero pT as in MadGraph LO events, the
face's azimuth-averaged kernel). The same calculation reproduces the archived
DarkQuest acceptance file — its r = 0.5 m disc, per χ — to 1–5% at every mass.

Relative to the paper's scan and acceptance file, at the same per-POT
normalisation (σ / a_geo / N_DY):

| | m_χ ≲ 1 GeV | 2–3 GeV | 5 GeV | 8 GeV |
|---|---|---|---|---|
| DarkQuest, iron | +8% / +33–42% / +44–53% | 0–9% / +35–44% / +35–55% | −1% / +20% / +19% | — |
| SHiP, molybdenum | +3% / +7–14% / +10–17% | +8% / +15–21% / +24–31% | −2% / +17% / +15% | −13% / +24% / +8% |

- **The acceptance combines detector and PDFs.** The 1×1 m face takes
  1.21–1.24× the disc's acceptance. SHiP's archived file was already a 1 m²
  face below 7 GeV (see below), so only the PDF part changes there. The PDFs
  add +10–20% below 3 GeV: a free-proton beam and the target's neutrons make
  the pairs more forward.
- **The cross section moves for other reasons than the material.** Fe → Mo
  changes the per-nucleon σ by < 2% below 3 GeV (−8% at 8 GeV). The rest comes
  from the paper's MadGraph setup, which gave the beam proton the iron
  nucleus's PDF and left out the target's neutrons.
- **Net effect.** With the normalisation, the default Drell-Yan is ×1.23–1.61
  the paper's for DarkQuest and ×1.30–1.60 for SHiP. The ε limit is up to 10%
  (DarkQuest, 5 GeV) and 13% (SHiP, 8 GeV) stronger where Drell-Yan dominates.

**Other modes.**
- `--fidelity legacy` keeps the paper's scan, its acceptance file (and so the
  paper's disc) and 13 mb.
- A config that sets `data.dy_cross` itself keeps its own scan and
  acceptance file, and so does a run that regenerates Drell-Yan
  (`--regenerate`, `mcpsim generate dy`): MadGraph's cross sections with the
  iron nPDF for both beams, and its acceptance for your detector.
- A target or beam energy without a table falls back to the preset's iron
  scan, and says so. To get a table, add the material to
  `physics/constants.TARGET_MATERIALS` and rerun the generator.

**What the archived scans are** (reconstructed by reproducing them to
0.1–0.7% with an independent LO convolution, `mcpsim-output/dy_reproduction/`):
`q q~ > chi chi~ / z` with q = u d s c at LO, the iron bound-proton nPDF
nCTEQ15_56_26 (LHAPDF 100600) for **both** beams (stock MadGraph takes a single
`lhaid`, so the beam proton is also an iron-bound proton), μ_F = μ_R = M(χχ̄),
a **pair-mass cut M(χχ̄) ≥ 2 GeV**, α = 1/127.9 from the UFO model, and the
exact fixed-target √s — the paper used them for DarkQuest *and* SHiP (an
approximation confirmed by the authors). `generation/madgraph.py` uses these
settings and counts the acceptance per χ, like the archived files. It
generates both parton orderings (`q q~` and `q~ q`: MadGraph adds the mirror
process only when asked, and σ halves without it), in the beam-nucleon
centre of mass with massless beams, and boosts the events to the lab itself:
with the target at rest MadGraph writes the events in a lab frame rebuilt
from a massive beam of energy E_parton/x, off at these √ŝ ≈ 2 GeV, which
pushed large-x pairs forward and raised the acceptance by 3.4%. Below
m_χ ≈ 1 GeV the 2 GeV cut makes σ
flat (~499 pb for DarkQuest in the paper), leaving the low-mass region to the
meson channels.

**What the archived acceptance files are** (from the exact LO per-χ
acceptance): the probability that *one* χ of the pair enters the aperture,
with the pair at zero pT (MadGraph LO events). DarkQuest's matches the
r = 0.5 m disc at 40 m to 1–5%. SHiP's mixes two geometries: up to 7.04 GeV
(the 199 masses it shares with DarkQuest) a **1 m² face** at 100 m — a 1×1 m
square fits to 0.98–1.02, but LO cannot tell it from an equal-area disc —
20–28% more than the paper's r = 0.5 m disc; the 7.4–13 GeV extension is the
r = 0.5 m disc (1.00–1.04). Hence an 11% step in a_geo between the two. The
size-labelled variants in the data (`ageo_dy_ship_0.5m_final`,
`ageo_dy_dark_0.5m`, `…_1m…`) fit 1×1 m and 2×2 m faces best, counting events
with at least one χ instead, so those labels are half-widths. See the caveats
below.

The **original UFO model is recovered** — the paper's Zenodo record
([10.5281/zenodo.18330380](https://zenodo.org/records/18330380)) ships
`Minimal_MCP.zip`, the FeynRules SM+χ model behind the scans (χ pdg 31, mass
`Mchi` in MASS[31], millicharge `qX` in FRBlock[1]; couples to γ *and* Z,
though the Z is irrelevant at these masses). Point `MCPSIM_MCP_MODEL` at its
unzipped directory for exact reruns; the in-repo `models/mcp_ufo` is the
minimal QED-only recreation with the same χ pdg. See [data.md](data.md) for
the open provenance items.
By default the DY acceptance is the tables' LO acceptance of the **preset's
face**. For a custom or off-axis detector it is recomputed at LO for that
detector, with the tables' PDFs, when `MCPSIM_PDFSETS` names a directory of
the nCTEQ15 LHAPDF grids and `parton` is installed; the result is cached, so
later runs need neither. Without them the run skips Drell-Yan and says why:
the preset face's acceptance is orders of magnitude too large for a small far
face such as FLAME's. A scan's acceptance file (`data.dy_ageo`, or
`--fidelity legacy`) made for another detector is skipped the same way, since
it is never recomputed.

## Fidelity: corrected vs legacy

| Knob | legacy | corrected (default) | size of the difference |
|---|---|---|---|
| Dalitz I₃ constant | 3.14 | π | +0.051% on every Dalitz yield |
| Dalitz sampler | θ_V flat in [0,π], χ built about ẑ, unrotated boost (the C++ quirks) | isotropic γ*, χ about the boost axis | ≤±8% a_geo at 4 mrad, ≲1% on ε |
| brem bar-array acceptance | equal-area cone cut | exact face kernel + sub-binning | geometry-dependent, small on-axis |
| 2-body BW mother mass | identical in both (matches the C++) | — | (5–7% ρ a_geo vs naive fixed-pole; informational) |
| Drell-Yan per-nucleon normalisation | 13 mb (iron) for every target | σ_inel(p-target)/A | DY ×1.035 DarkQuest, ×1.22 SHiP |
| DarkQuest / SHiP detector | the paper's r = 0.5 m disc (`legacy_detector`) and its archived acceptance files | a 1×1 m face (20 × 20 bars of 5 cm, the brem grids' layout); meson acceptance recomputed for it (Python engine, cached), Drell-Yan from the tables | meson acceptance +17–33%; ε 1–3% stronger where mesons dominate; brem unchanged (see the caveats) |
| Drell-Yan | the paper's scan (iron bound proton for both beams, DarkQuest and SHiP alike) and acceptance file | per-material table (free proton × nucleus) with the LO acceptance of the face | with the rows above: DY ×1.23–1.61 DarkQuest, ×1.30–1.60 SHiP; ε up to 10% / 13% stronger where DY dominates ([above](#drell-yan)) |

Each row except the mother-mass one is a field of `config.FIDELITY_RULES`, the
one place the code tells the two modes apart. `--fidelity legacy` at preset
geometry reproduces the frozen golden tables bit-for-bit (`pytest -m golden`).

## Known physics caveats

Every approximation that can move a curve, in one place. "Run" means the run
prints a `!` line when the caveat applies; the rest are documented only.

| Caveat | Effect | Handling |
|---|---|---|
| **Υ has no acceptance scan** — it borrows this run's low-mass J/ψ acceptance, flat in mass, beyond the J/ψ scan's 1.525 GeV end; its per-POT rate cannot be checked (that PYTHIA run was never made) | channel is ~10⁻⁴ of the J/ψ; it matters only for m_χ ≈ 1.55–4.73 GeV, where its sole competitor is Drell-Yan | run (when borrowed or skipped); [section above](#acceptance-convention-the-2) |
| **Meson multiplicities are per hard-QCD trial** (`HardQCD:all`, `pTHatMin = 2`): entries/trials of a biased sample, not per inelastic collision | J/ψ 4.1×10⁻⁵/POT at 120 GeV is ~10× what σ(J/ψ) ≈ 50–80 nb/nucleon at √s ≈ 15 GeV implies (≈3×10⁻⁶ per p-C interaction — an estimate, not yet checked against data); the 0.5–1.5 GeV plateau would be ~1.3× weaker in ε | documented |
| **Dalitz decays are point-like** (Kroll–Wada rate, no transition form factor) | negligible for π⁰; for η the form factor (Λ ≈ 0.72 GeV) raises the rate by tens of percent to ~2× at m_χ ≈ 0.1–0.25 GeV, so η yields there are conservative | documented |
| **Brem grids carry their generating detector's acceptance** (single track or shared bar on a 1×1 m face at 40 / 100 m, with the first χ always at azimuth 0) | they resolve the face only out to its half-width, `data.brem_max_theta_mrad` (DarkQuest/FLAME grids 12.5 mrad, SHiP 5 mrad): brem through the corners of the presets' own face is missing (≲ 20% of it), and a far off-axis face collapses at low mass; below that angle, both-χ hits are undercounted by up to ×2 | run |
| **Brem target** — grids contain σ_pA of the nucleus they were made for | rescaled to the configured target from the grid header | run (when rescaled) |
| **Brem low-mass rise** (~1/m_χ² below ~30 MeV) comes from the longitudinal H/(2zk²) term of the massive-photon QRA kernel, i.e. k^μk^ν contracted with a proton current the QRA does not conserve (energy is not conserved at the p→p′γ* vertex). The Dawson-subtracted polarisation sum (Foroughi-Abari, Reimitz & Ritz, PRD 112 (2025) 015030; Kling, Reimitz & Ritz, PRD 113 (2026) 015008) gives a ~flat yield | full QRA / Dawson at a FLAME-like face: ~700× at 1 MeV, ~9× at 10 MeV, ×1.5 at 50 MeV, <20% above 100 MeV. The grids use the full QRA (the paper's choice), the upper of the two | documented |
| **Brem eVMD resonances** are also in the PYTHIA ρ/ω channels | double counting in m ≈ 0.2–0.5 GeV; negligible where brem ≫ ρ/ω | documented ([above](#proton-bremsstrahlung)) |
| **Brem θ binning** (0.045 dex) is smeared uniformly in log θ for small faces | ±15% sawtooth in brem vs off-axis angle (≈2% in ε) | documented |
| **Drell-Yan material**: tables exist for iron, molybdenum, tungsten, carbon at 120 and 400 GeV | another target or beam energy falls back to the preset's iron scan, whose per-nucleon nPDF difference from the target is then missing (up to ~10% for m_χ ≲ 3 GeV, ~25% near the kinematic limit) | run (on fallback, or when `data.dy_material` ≠ target) |
| **Drell-Yan is LO** (no K-factor; NLO is typically ×1.5–2 at fixed-target energies) | conservative | documented |
| **Drell-Yan pair-mass cut M(χχ̄) ≥ 2 GeV** (archived scans and tables) | below m_χ ≈ 1 GeV DY counts only pairs above 2 GeV; lower pair masses are left to the meson channels (and pQCD with PDFs below their Q_min would not be reliable there) | documented |
| **Drell-Yan PDF and α choices**: nCTEQ15, α = 1/127.9 (the archived scans' value) | the free-proton PDF family moves σ by 3–25% for m_χ ≲ 3 GeV and up to ×1.8 near the kinematic limit (MSTW2008lo vs nCTEQ15); α at the DY scale (~1/133) would give −8%, α(0) = 1/137 −13% | documented |
| **Drell-Yan counts one χ per pair**: the acceptance files hold the per-χ hit probability and N_DY has no factor 2, while the meson channels count both χ (2 × per-χ) | counting both would double N_DY (ε up to 8% stronger in the light-yield-limited regime, more where P → 1) | documented — a convention to settle |
| **Drell-Yan acceptance is LO collinear** (pair pT = 0, as in the MadGraph LO events) | real pairs carry ⟨pT⟩ ~ 1 GeV, comparable to the transverse momentum that already takes a χ out of the 5–12.5 mrad cones; not in the files nor the reweighting (unquantified) | documented |
| **SHiP's archived Drell-Yan acceptance mixes geometries** (`--fidelity legacy`): a 1 m² face (1×1 m square or equal-area disc) up to 7.04 GeV, the paper's r = 0.5 m disc above 7.4 GeV | 20–28% more acceptance than the disc below 7 GeV; an 11% step in a_geo at 7.0–7.4 GeV; the paper's SHiP ε is ~3% too strong at 1.6–5 GeV if the disc was meant (6.5% at 7 GeV). The default uses the LO acceptance of the 1×1 m face at every mass | documented ([above](#drell-yan)) |
| **Drell-Yan acceptance of a custom or off-axis detector** | recomputed at LO for that detector when `MCPSIM_PDFSETS` and `parton` are available (cached); otherwise Drell-Yan is skipped, since the preset face's would be orders of magnitude too large for FLAME | run |
| **Zero-background threshold** (3.09 events) and borrowed `n_gamma` anchors | see [the detection model](#detection-model-and-the-exclusion-threshold) | documented |
| **Limit floor** — where the signal already passes threshold at `charge_min`, the tabulated ε is the scan's bottom, not the limit | limit stronger than shown | run |

## Known provenance gaps (inherited)

- The 120 GeV PYTHIA samples were generated with `HardQCD:all = on,
  PhaseSpace:pTHatMin = 2` — a
  hard-QCD-biased sample, not min-bias; the same bias underlies the published
  DarkQuest acceptances.
- DY: see [data.md](data.md) (`_iron` tags on the molybdenum SHiP preset,
  nCTEQ15/nCTEQ naming, the settings reconstructed from the scans, the
  acceptance files' conventions).
