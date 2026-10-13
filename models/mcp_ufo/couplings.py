# Photon couplings: quark charges and the millicharge EPS * e.

try:  # imported as the mcp_ufo package (mcpsim, the tests)
    from .object_library import all_couplings, Coupling
except ImportError:  # module by module (MadGraph's internal UFO scripts)
    from object_library import all_couplings, Coupling

GC_a_up = Coupling(name="GC_a_up", value="(2./3.)*ee*complex(0,1)",
                   order={"QED": 1})

GC_a_down = Coupling(name="GC_a_down", value="-(1./3.)*ee*complex(0,1)",
                     order={"QED": 1})

GC_a_chi = Coupling(name="GC_a_chi", value="EPS*ee*complex(0,1)",
                    order={"QED": 1})
