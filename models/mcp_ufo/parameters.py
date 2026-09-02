# Parameters of the millicharged-fermion QED model.
#
# External (param_card blocks):
#   SMINPUTS:  aEWM1 (1/alpha_EM), aS (alpha_s; carried for PDF machinery,
#              no QCD vertices in this model)
#   MCPINPUTS: EPS — the millicharge in units of e (the chi-photon coupling
#              is EPS * e; cross sections scale as EPS^2)
#   MASS:      MCHI (pdg 9000001)

from .object_library import Parameter

aEWM1 = Parameter(name="aEWM1", nature="external", type="real",
                  value=127.9, texname="\\text{aEWM1}",
                  lhablock="SMINPUTS", lhacode=[1])

aS = Parameter(name="aS", nature="external", type="real",
               value=0.1184, texname="\\alpha _s",
               lhablock="SMINPUTS", lhacode=[3])

EPS = Parameter(name="EPS", nature="external", type="real",
                value=1.0, texname="\\epsilon",
                lhablock="MCPINPUTS", lhacode=[1])

MCHI = Parameter(name="MCHI", nature="external", type="real",
                 value=0.1, texname="M_{\\chi}",
                 lhablock="MASS", lhacode=[31])

# -- internal ---------------------------------------------------------------
aEW = Parameter(name="aEW", nature="internal", type="real",
                value="1./aEWM1", texname="\\alpha _{\\text{EW}}")

ee = Parameter(name="ee", nature="internal", type="real",
               value="2*cmath.sqrt(aEW)*cmath.sqrt(cmath.pi)",
               texname="e")

G = Parameter(name="G", nature="internal", type="real",
              value="2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)",
              texname="G")

ZERO = Parameter(name="ZERO", nature="internal", type="real",
                 value="0.0", texname="0")
