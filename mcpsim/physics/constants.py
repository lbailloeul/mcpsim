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

# Per-POT meson multiplicities (c_meson) from PYTHIA, by beam energy.
#
# Each value is (entries in that beam's parent sample) / (trials that generated
# it), and reproduces the published table in arXiv:2512.11027 to the quoted
# precision. The trial counts are not stored in the sample files -- they live in
# the generator's run script (pi0 5e4, eta 2e5, rho 1e6, omega 1e6, phi 1e7,
# jpsi 1e8), so the two have to be kept in step by hand.
#
# The upsilon entries cannot be checked this way: that generation was never run
# (commented out in the run script), which is also why the upsilon channel
# has to borrow the J/psi acceptance (see UPSILON_FAMILIES below).
#
# These are per-beam-energy: change the beam and every value changes. mcpsim
# will not extrapolate -- an energy outside the tabulated set (matched within
# 2%) raises, and you must supply data.c_meson yourself. Note that --regenerate
# does NOT fill this in for you: it produces new parent samples but never counts
# them, so the multiplicity for a new energy still has to be computed by hand as
# entries / trials.
#
# The 120 GeV J/psi is carried at full precision (4087 parents / 1e8 trials);
# the published plotting scripts rounded it to 4.0e-5. Do not "correct" it back.
C_MESON_BY_BEAM: Dict[str, Dict[str, float]] = {
    # DarkQuest / SpinQuest, 120 GeV protons.
    "120": {
        "pi0": 4.7, "eta": 0.53, "rho": 0.61, "omega": 0.61,
        "phi": 2.2e-2, "jpsi": 4.087e-5, "upsilon": 2.5e-9,
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

    Linear scaling is an energy-deposition argument (dE/dx * L) and holds near
    the anchor length. It says nothing about light *collection*, which falls off
    over metre-scale bars through bulk attenuation, so do not read across a
    large change in length -- the per-metre yields of two materials anchored at
    very different lengths are not comparable for that reason.
    """
    name: str
    n_gamma_ref: float
    length_ref_m: float


SCINTILLATORS: Dict[str, ScintillatorSpec] = {
    # 1.5 m plastic bar (FLAME/milliQan-style): 2.5e5 PE at eps = 1.
    "plastic": ScintillatorSpec("plastic", 2.5e5, 1.5),
    # CeBr: 5.0e6 PE at eps = 1 for a 1.5 m bar. The LANSCE 12-bar
    # demonstrator's 5 cm crystals scale down from this to 1.67e5.
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

# Experiment families for which the upsilon channel is computed.
#
# The upsilon is the one meson with no acceptance scan of its own. It is simply
# produced too rarely for a parent sample to have been built: at 120 GeV
# upsilons come out some 1.6e4 times less often than J/psi (see C_MESON_BY_BEAM
# above), and even the J/psi sample holds only a few thousand events. Building
# a comparable upsilon sample is a PYTHIA campaign in its own right, and
# attempts at one have not been practical.
#
# So the upsilon borrows the J/psi's acceptance instead -- specifically the
# first (low-mass) value of the J/psi acceptance computed for THAT run, held
# flat across the mass grid. A preset geometry reads it from the archived scan;
# an off-axis or custom geometry gets the value the Python engine just computed
# for that detector. Without a J/psi acceptance there is nothing to borrow and
# the channel is skipped (model._upsilon_ageo).
#
# This set gates the channel: a family not listed here has no established basis
# for the substitution and skips it. FLAME relies on that -- its 1 km bar array
# is far enough from the DarkQuest/SHiP geometries that the borrow was judged
# unsound.
#
# What stays unvalidated: the J/psi acceptance is not flat in mass, and its scan
# stops at m_chi = 1.525 GeV (the J/psi's own kinematic edge) while upsilon
# decays stay open to m_chi = 4.73 GeV, so past roughly the first third of the
# upsilon's range the borrowed value covers territory no scan has reached. The
# channel is worth ~1e-4 of the J/psi yield and has never shifted a limit.
UPSILON_FAMILIES = frozenset({"darkquest", "ship"})
