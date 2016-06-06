from frules.expressions import Expression as E
from frules.rules import Rule as R
from frules.expressions import ltrapezoid, trapezoid, rtrapezoid

# lab expression
near = E(ltrapezoid(1.8, 2.4), "near")
far = E(trapezoid(1.8, 2, 3.3, 3.5), "far")
far_away = - (near & far)

# neighbours expression
few = E(ltrapezoid(0.2, 0.3), "few")
rather = E(trapezoid(0.2, 0.3, 0.5, 0.6), "rather")
a_lot = - (few & rather)

is_closed = R(lab = near)
is_noised = R(neighbours = a_lot)

a = {"lab" : 1.9}
print is_closed.eval(**a)
