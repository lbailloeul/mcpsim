# mcp_ufo: millicharged Dirac fermion QED model for MadGraph5_aMC@NLO.
#
# NOTE — the ORIGINAL model is recovered: the paper's Zenodo record
# (10.5281/zenodo.18330380) ships Minimal_MCP.zip, the FeynRules-generated
# full SM + chi UFO the archived DY scans were produced with (chi pdg 31,
# mass 'Mchi' in MASS[31], millicharge 'qX' in FRBlock[1]; couples to gamma
# AND Z). For exact reproduction of the archived scans use that model
# (set MCPSIM_MCP_MODEL to the unzipped Minimal_MCP directory). This in-repo
# model is the minimal QED-only recreation — same chi pdg (31) and physics at
# DY scales far below M_Z — kept as a self-contained, structurally-tested
# fallback.
#
# Content: five massless quarks + gluon + photon + a Dirac fermion chi
# (pdg 31, mass MCHI, photon coupling EPS * e). QED vertices only —
# intended for LO qq_bar -> gamma* -> chi chi_bar (Drell-Yan mCP production).
#
# Usage:
#   export MCPSIM_MCP_MODEL=/path/to/mcpsim/models/mcp_ufo   # or rely on the
#   default: mcpsim.generation.madgraph falls back to this directory.
#   In MG5:  import model /path/to/models/mcp_ufo
#            generate p p > chi chi~
#
# Validation recipe (needs mg5_aMC + LHAPDF; see docs/physics.md):
#   1. Muon-pair benchmark: temporarily set EPS=1, MCHI=0.105 and compare
#      sigma(p p > chi chi~) against p p > mu+ mu- in the SM model at the
#      same cuts/PDF — they must agree at the few-per-mil level (identical
#      matrix elements at these settings).
#   2. Archived-scan point: reproduce one mass point of
#      dy_cross_darkquest_nCTEQ15_iron.txt with the nCTEQ15 iron PDF at
#      120 GeV fixed-target settings.
# The analytic partonic cross section for cross-checks is
# mcpsim.physics.dy_lo.sigma_partonic_pb.

from . import (coupling_orders, couplings, function_library, lorentz,  # noqa: F401
               object_library, parameters, particles, vertices)

all_particles = object_library.all_particles
all_parameters = object_library.all_parameters
all_vertices = object_library.all_vertices
all_couplings = object_library.all_couplings
all_lorentz = object_library.all_lorentz
all_orders = object_library.all_orders
all_functions = object_library.all_functions
all_decays = object_library.all_decays

__author__ = "mcpsim (recreated; original DY model not preserved)"
__version__ = "1.0"
