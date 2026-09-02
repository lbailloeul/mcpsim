# Particle content: the five light quarks + gluon + photon (enough for a
# proton beam) and the millicharged Dirac fermion chi (pdg 9000001).
# Quarks are massless (5-flavour PDF convention). The gluon is present so
# the standard `p` multiparticle definition loads; there are NO QCD vertices
# in this model, so gluon-initiated subprocesses simply have no diagrams at
# the qq_bar -> gamma* -> chi chi_bar order.

from . import parameters as Param
from .object_library import Particle

a = Particle(pdg_code=22, name="a", antiname="a", spin=3, color=1,
             mass=Param.ZERO, width=Param.ZERO, texname="a", antitexname="a",
             charge=0)

g = Particle(pdg_code=21, name="g", antiname="g", spin=3, color=8,
             mass=Param.ZERO, width=Param.ZERO, texname="g", antitexname="g",
             charge=0)

u = Particle(pdg_code=2, name="u", antiname="u~", spin=2, color=3,
             mass=Param.ZERO, width=Param.ZERO, texname="u", antitexname="u~",
             charge=2. / 3.)
u__tilde__ = u.anti()

c = Particle(pdg_code=4, name="c", antiname="c~", spin=2, color=3,
             mass=Param.ZERO, width=Param.ZERO, texname="c", antitexname="c~",
             charge=2. / 3.)
c__tilde__ = c.anti()

d = Particle(pdg_code=1, name="d", antiname="d~", spin=2, color=3,
             mass=Param.ZERO, width=Param.ZERO, texname="d", antitexname="d~",
             charge=-1. / 3.)
d__tilde__ = d.anti()

s = Particle(pdg_code=3, name="s", antiname="s~", spin=2, color=3,
             mass=Param.ZERO, width=Param.ZERO, texname="s", antitexname="s~",
             charge=-1. / 3.)
s__tilde__ = s.anti()

b = Particle(pdg_code=5, name="b", antiname="b~", spin=2, color=3,
             mass=Param.ZERO, width=Param.ZERO, texname="b", antitexname="b~",
             charge=-1. / 3.)
b__tilde__ = b.anti()

# The millicharged fermion. `charge` is the electric charge in units of e at
# the reference EPS = 1; the actual coupling strength is EPS * e via GC_chi.
# pdg 31 matches the original Minimal_MCP UFO and the PYTHIA samples' mcp id.
chi = Particle(pdg_code=31, name="chi", antiname="chi~", spin=2, color=1,
               mass=Param.MCHI, width=Param.ZERO, texname="\\chi",
               antitexname="\\bar{\\chi}", charge=1)
chi__tilde__ = chi.anti()
