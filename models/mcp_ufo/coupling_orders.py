try:  # imported as the mcp_ufo package (mcpsim, the tests)
    from .object_library import all_orders, CouplingOrder
except ImportError:  # module by module (MadGraph's internal UFO scripts)
    from object_library import all_orders, CouplingOrder

QCD = CouplingOrder(name="QCD", expansion_order=99, hierarchy=1)
QED = CouplingOrder(name="QED", expansion_order=99, hierarchy=2)
