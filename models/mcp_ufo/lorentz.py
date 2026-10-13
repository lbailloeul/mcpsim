# Lorentz structures. One vector current is all this model needs.

try:  # imported as the mcp_ufo package (mcpsim, the tests)
    from .object_library import all_lorentz, Lorentz
except ImportError:  # module by module (MadGraph's internal UFO scripts)
    from object_library import all_lorentz, Lorentz

FFV1 = Lorentz(name="FFV1", spins=[2, 2, 3], structure="Gamma(3,2,1)")
