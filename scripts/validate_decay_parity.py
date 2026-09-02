#!/usr/bin/env python3
"""DEPRECATED: the decay parity harness now lives in mcpsim/validation/.

The three tests (A: 2-body vs decayVectorMeson.cc; B: Dalitz vs
lanl_decayPion_12bar.cc; C: corrected-vs-legacy fidelity delta) live in
mcpsim/validation/parity.py and run against WORK_DIR/validation scratch —
the old hardcoded session scratchpad path is gone.

    pytest -m parity                            # full harness (needs ROOT + samples)
    python -m mcpsim.validation.parity --quick  # one mass per test
"""

import sys

print(__doc__)
sys.exit(0 if "--help" in sys.argv or "-h" in sys.argv else 1)
