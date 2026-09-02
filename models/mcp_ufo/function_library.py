# Standard UFO function library (complex math helpers used in parameter
# expressions).

import cmath

from .object_library import Function

complexconjugate = Function(name="complexconjugate",
                            arguments=("z",),
                            expression="z.conjugate()")

re = Function(name="re", arguments=("z",), expression="z.real")

im = Function(name="im", arguments=("z",), expression="z.imag")

sec = Function(name="sec", arguments=("z",), expression="1./cmath.cos(z)")

asec = Function(name="asec", arguments=("z",), expression="cmath.acos(1./z)")

csc = Function(name="csc", arguments=("z",), expression="1./cmath.sin(z)")

acsc = Function(name="acsc", arguments=("z",), expression="cmath.asin(1./z)")

cot = Function(name="cot", arguments=("z",), expression="1./cmath.tan(z)")
