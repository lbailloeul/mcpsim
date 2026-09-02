#!/usr/bin/env python3
"""DEPRECATED: the FLAME off-axis study is now a first-class mcpsim preset.

This standalone script has been promoted into the package:
  * decay engine        -> mcpsim/generation/pydecay.py
  * azimuthal kernel    -> mcpsim/geometry.py (face_hit_fraction, generalized)
  * off-axis brem       -> mcpsim/physics/brem.py (brem_yield_offaxis)
  * contour solver      -> mcpsim/physics/sensitivity.py (exclusion_contour)
  * geometry & exposure -> mcpsim/presets/flame.yaml

Reproduce (and extend) the original study with:

    mcpsim run --preset flame --offaxis 4,6.5,9,11.5,14

Add `--channels meson_decay,brem` for the brem overlay (mind the eVMD
resonance double-counting note in the preset), `--fidelity legacy` for the
legacy C++ sampler behaviour, `--quick` for a smoke run.

The original script (kernel validation history, run_meta conventions) is
preserved in git history prior to this shim.
"""

import sys

print(__doc__)
sys.exit(0 if "--help" in sys.argv or "-h" in sys.argv else 1)
