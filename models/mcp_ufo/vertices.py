# Vertices: q q~ a for the five flavours, and chi chi~ a.

from . import couplings as C
from . import lorentz as L
from . import particles as P
from .object_library import Vertex

V_ua = Vertex(name="V_ua", particles=[P.u__tilde__, P.u, P.a],
              color=["Identity(1,2)"], lorentz=[L.FFV1],
              couplings={(0, 0): C.GC_a_up})

V_ca = Vertex(name="V_ca", particles=[P.c__tilde__, P.c, P.a],
              color=["Identity(1,2)"], lorentz=[L.FFV1],
              couplings={(0, 0): C.GC_a_up})

V_da = Vertex(name="V_da", particles=[P.d__tilde__, P.d, P.a],
              color=["Identity(1,2)"], lorentz=[L.FFV1],
              couplings={(0, 0): C.GC_a_down})

V_sa = Vertex(name="V_sa", particles=[P.s__tilde__, P.s, P.a],
              color=["Identity(1,2)"], lorentz=[L.FFV1],
              couplings={(0, 0): C.GC_a_down})

V_ba = Vertex(name="V_ba", particles=[P.b__tilde__, P.b, P.a],
              color=["Identity(1,2)"], lorentz=[L.FFV1],
              couplings={(0, 0): C.GC_a_down})

V_chia = Vertex(name="V_chia", particles=[P.chi__tilde__, P.chi, P.a],
                color=["1"], lorentz=[L.FFV1],
                couplings={(0, 0): C.GC_a_chi})
