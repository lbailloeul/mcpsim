"""Heavy generation backends (used only with --regenerate).

These wrap the existing C++ generators (PYTHIA, Burman-Smith, ROOT decay), the
VEGAS bremsstrahlung integrator, and MadGraph for Drell-Yan. Each backend
performs an environment check and degrades gracefully when its toolchain is not
installed, so fast mode never imports them.
"""
