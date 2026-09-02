# Standard UFO object library (UFO format 1.0), trimmed to what this model
# uses. Recreated for the mcpsim mCP model; API-compatible with the
# FeynRules-generated original.

all_particles = []
all_parameters = []
all_vertices = []
all_couplings = []
all_lorentz = []
all_orders = []
all_functions = []
all_decays = []


class UFOError(Exception):
    pass


class UFOBaseClass(object):
    require_args = []

    def __init__(self, *args, **kwargs):
        for name, value in zip(self.require_args, args):
            setattr(self, name, value)
        for name, value in kwargs.items():
            setattr(self, name, value)

    def get(self, name):
        return getattr(self, name)

    def set(self, name, value):
        setattr(self, name, value)

    def get_all(self):
        return self.__dict__

    def __str__(self):
        return getattr(self, "name", object.__repr__(self))

    nice_string = __str__


class Particle(UFOBaseClass):
    require_args = ["pdg_code", "name", "antiname", "spin", "color", "mass",
                    "width", "texname", "antitexname", "charge"]

    def __init__(self, pdg_code, name, antiname, spin, color, mass, width,
                 texname, antitexname, charge, line=None, propagating=True,
                 goldstoneboson=False, **options):
        args = (pdg_code, name, antiname, spin, color, mass, width, texname,
                antitexname, float(charge))
        UFOBaseClass.__init__(self, *args, **options)
        global all_particles
        all_particles.append(self)
        self.propagating = propagating
        self.goldstoneboson = goldstoneboson
        self.selfconjugate = (name == antiname)
        self.line = line if line is not None else self.find_line_type()

    def find_line_type(self):
        spin, color = self.spin, self.color
        if spin == 1:
            return "dashed"
        if spin == 2:
            if not self.selfconjugate:
                return "straight"
            if color == 1:
                return "swavy"
            return "scurly"
        if spin == 3:
            if color == 1:
                return "wavy"
            return "curly"
        if spin == 5:
            return "double"
        if spin == -1:
            return "dotted"
        return "dashed"

    def anti(self):
        if self.selfconjugate:
            raise UFOError("%s is its own antiparticle" % self.name)
        outdic = {}
        for k, v in self.__dict__.items():
            if k not in self.require_args_all():
                outdic[k] = -v if isinstance(v, (int, float)) and k == "charge" else v
        if self.color in (3, 6):
            newcolor = -self.color
        else:
            newcolor = self.color
        return Particle(-self.pdg_code, self.antiname, self.name, self.spin,
                        newcolor, self.mass, self.width, self.antitexname,
                        self.texname, -self.charge, self.line,
                        self.propagating, self.goldstoneboson)

    def require_args_all(self):
        return ["pdg_code", "name", "antiname", "spin", "color", "mass",
                "width", "texname", "antitexname", "charge", "line",
                "propagating", "goldstoneboson"]


class Parameter(UFOBaseClass):
    require_args = ["name", "nature", "type", "value", "texname"]

    def __init__(self, name, nature, type, value, texname, lhablock=None,
                 lhacode=None):
        args = (name, nature, type, value, texname)
        UFOBaseClass.__init__(self, *args)
        global all_parameters
        all_parameters.append(self)
        if (lhablock is None or lhacode is None) and nature == "external":
            raise UFOError("external parameter %s needs lhablock/lhacode" % name)
        self.lhablock = lhablock
        self.lhacode = lhacode


class Vertex(UFOBaseClass):
    require_args = ["name", "particles", "color", "lorentz", "couplings"]

    def __init__(self, name, particles, color, lorentz, couplings, **opt):
        args = (name, particles, color, lorentz, couplings)
        UFOBaseClass.__init__(self, *args, **opt)
        global all_vertices
        all_vertices.append(self)


class Coupling(UFOBaseClass):
    require_args = ["name", "value", "order"]

    def __init__(self, name, value, order, **opt):
        args = (name, value, order)
        UFOBaseClass.__init__(self, *args, **opt)
        global all_couplings
        all_couplings.append(self)


class Lorentz(UFOBaseClass):
    require_args = ["name", "spins", "structure"]

    def __init__(self, name, spins, structure="external", **opt):
        args = (name, spins, structure)
        UFOBaseClass.__init__(self, *args, **opt)
        global all_lorentz
        all_lorentz.append(self)


class CouplingOrder(object):
    def __init__(self, name, expansion_order, hierarchy, perturbative_expansion=0):
        global all_orders
        all_orders.append(self)
        self.name = name
        self.expansion_order = expansion_order
        self.hierarchy = hierarchy
        self.perturbative_expansion = perturbative_expansion


class Function(object):
    def __init__(self, name, arguments, expression):
        global all_functions
        all_functions.append(self)
        self.name = name
        self.arguments = arguments
        self.expr = expression

    def __call__(self, *opt):
        for i, arg in enumerate(self.arguments):
            exec("%s = %s" % (arg, opt[i]))
        return eval(self.expr)


class Decay(UFOBaseClass):
    require_args = ["particle", "partial_widths"]

    def __init__(self, particle, partial_widths, **opt):
        args = (particle, partial_widths)
        UFOBaseClass.__init__(self, *args, **opt)
        global all_decays
        all_decays.append(self)
        particle.partial_widths = partial_widths
