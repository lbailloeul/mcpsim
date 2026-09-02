"""Fundamental constants, meson data and target-material table.

Values are consolidated from DrawProductionDarkQuest.py / DrawProductionSHIP.py /
DrawProbability.py / plot_lanl_12bar_sensitivity.py, which previously each
hardcoded their own copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# -- Fundamental constants --------------------------------------------------
ALPHA_EM = 1.0 / 137.0
M_PROTON = 0.9383          # GeV
M_ELECTRON = 0.00051       # GeV
N_AVOGADRO = 6.02214076e23  # /mol
PB_TO_CM2 = 1.0e-36        # 1 pb in cm^2

# Total inelastic proton-nucleon cross section normalising the Drell-Yan
# probability per interaction: N = N_POT * sigma_DY / sigma_inel * ageo.
# The legacy scripts encoded this as sigma_pb*1e-12 / 13e-3 — dimensionally
# mislabeled ("pb -> cm^2" over a "pb-scale" total) but numerically identical
# to sigma_pb / (13 mb in pb); see physics/drellyan.py and docs/physics.md.
SIGMA_PN_INELASTIC_MB = 13.0
MB_TO_PB = 1.0e9


@dataclass(frozen=True)
class MesonSpec:
    """A meson decay channel into a millicharged pair.

    channel:
      'dalitz'   -> M -> gamma chi chi-bar (3-body), uses I3 and an extra alpha.
      'two_body' -> M -> chi chi-bar       (2-body), uses I2.
    branching is the relevant SM branching ratio (gamma-gamma for Dalitz parents,
    e+e- for the direct vector-meson decays) that the mCP rate is scaled from.
    """

    name: str
    mass: float          # GeV
    branching: float
    channel: str         # 'dalitz' | 'two_body'
    label: str           # matplotlib legend label


# Ordered as the legacy production plots draw them.
MESONS: Dict[str, MesonSpec] = {
    "pi0": MesonSpec("pi0", 0.135, 0.98, "dalitz", r"$\pi^{0}\!\to\!\gamma\chi\bar{\chi}$"),
    "eta": MesonSpec("eta", 0.548, 0.39, "dalitz", r"$\eta\!\to\!\gamma\chi\bar{\chi}$"),
    "jpsi": MesonSpec("jpsi", 3.1, 0.05971, "two_body", r"$J/\psi\!\to\!\chi\bar{\chi}$"),
    "upsilon": MesonSpec("upsilon", 9.46, 0.0238, "two_body", r"$\Upsilon\!\to\!\chi\bar{\chi}$"),
    "rho": MesonSpec("rho", 0.775, 4.72e-5, "two_body", r"$\rho\!\to\!\chi\bar{\chi}$"),
    "omega": MesonSpec("omega", 0.782, 7.28e-5, "two_body", r"$\omega\!\to\!\chi\bar{\chi}$"),
    "phi": MesonSpec("phi", 1.019, 2.95e-4, "two_body", r"$\phi\!\to\!\chi\bar{\chi}$"),
}

# Mesons accessible to the low-energy (Burman-Smith) generator. PYTHIA hard-QCD is
# needed for the heavier / vector states.
LOW_ENERGY_MESONS = ("pi0", "eta")

# Per-POT meson multiplicities (c_meson) from PYTHIA, by beam energy. These are
# *production* normalisations and cannot be derived without running the generator,
# so fast mode reuses these tabulated sets; an unknown beam energy needs
# --regenerate (PYTHIA reports the multiplicity directly).
C_MESON_BY_BEAM: Dict[str, Dict[str, float]] = {
    # DarkQuest / SpinQuest, 120 GeV protons.
    "120": {
        "pi0": 4.7, "eta": 0.53, "rho": 0.61, "omega": 0.61,
        "phi": 2.2e-2, "jpsi": 4.0e-5, "upsilon": 2.5e-9,
    },
    # SHiP, 400 GeV protons.
    "400": {
        "pi0": 7.5, "eta": 0.85, "rho": 1.0, "omega": 1.0,
        "phi": 4.1e-2, "jpsi": 8.3e-5, "upsilon": 5.5e-9,
    },
    # LANSCE 800 MeV (Burman-Smith). Only pi0/eta; eta ~ pi0/30 (see LANL scripts).
    "0.8": {"pi0": 0.115, "eta": 0.115 / 30.0},
}


@dataclass(frozen=True)
class ScintillatorSpec:
    """Light-yield anchor for deriving sensitivity.n_gamma from geometry.

    n_gamma_ref is the mean photoelectron count at eps = 1 for a full
    traversal of a bar of length_ref_m (GEANT4-derived group numbers);
    n_gamma scales linearly with the traversed bar length:
        n_gamma = n_gamma_ref * bar_length_m / length_ref_m.
    """
    name: str
    n_gamma_ref: float
    length_ref_m: float


SCINTILLATORS: Dict[str, ScintillatorSpec] = {
    # 1.5 m plastic bar (FLAME/milliQan-style): 2.5e5 PE at eps = 1.
    "plastic": ScintillatorSpec("plastic", 2.5e5, 1.5),
    # 1.5 m CeBr: 5.0e6 PE at eps = 1. (The LANSCE 12-bar preset's explicit
    # n_gamma = 5.0e6 is the proposal's own number for its 5 cm crystals and
    # is NOT derived from this anchor.)
    "cebr": ScintillatorSpec("cebr", 5.0e6, 1.5),
}


@dataclass(frozen=True)
class TargetMaterial:
    name: str
    Z: int
    A: float              # g/mol
    density_g_cm3: float
    interaction_length_cm: float


TARGET_MATERIALS: Dict[str, TargetMaterial] = {
    "iron": TargetMaterial("iron", 26, 55.845, 7.87, 16.8),
    "molybdenum": TargetMaterial("molybdenum", 42, 95.95, 10.2, 15.27),
    "tungsten": TargetMaterial("tungsten", 74, 183.84, 19.3, 9.95),
    "carbon": TargetMaterial("carbon", 6, 12.011, 2.0, 38.1),
}

# Geometric acceptance for the upsilon channel, one value per experiment family.
#
# The upsilon is the one meson with no acceptance scan of its own. It is simply
# produced too rarely for us to have built a parent sample: at 120 GeV upsilons
# come out some 1.6e4 times less often than J/psi (see C_MESON_BY_BEAM below),
# and even the J/psi sample we do have holds only 4132 events. Building an
# upsilon sample of comparable size is a PYTHIA campaign in its own right, and
# attempts at one have not been practical.
#
# So we use the J/psi acceptance in its place, taking the value from the
# low-mass end of the same family's scan and applying it at every mass:
#   darkquest 0.011357  from total_efficiency_output2body_decay-jsi.txt
#   ship      0.011338  from total_efficiency_output2body_decay-jsi-SHiP.txt
# Both scans really are flat there (over their first 68 and 109 rows), so the
# borrowed number is at least well defined.
#
# For a channel worth roughly 1e-4 of the J/psi yield this is a comfortable
# approximation, and it has never shifted a limit. Two things would be worth
# checking if an upsilon sample ever becomes available. The J/psi acceptance is
# not flat across the whole range -- the same scans rise to 0.019448 (darkquest)
# and 0.04961 (ship) by their last row. And they stop at m_chi = 1.525 GeV,
# essentially the J/psi's own kinematic edge, while upsilon decays stay open out
# to m_chi = 4.73 GeV, so beyond about the first third of the upsilon's range we
# are carrying a borrowed value into a region no scan has covered.
#
# Practical note: the substitution is written as its own branch on the upsilon
# in model.py (_add_meson_channels) rather than as a general fallback, so
# introducing a real scan later means registering the sample in DEFAULT_SAMPLES
# and removing that branch.
UPSILON_AGEO_DEFAULT = {"darkquest": 0.011357, "ship": 0.011338}
