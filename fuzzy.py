from frules.expressions import Expression as E
from frules.rules import Rule as R
from frules.expressions import ltrapezoid, trapezoid, rtrapezoid

# lab expression
near = E(ltrapezoid(1.8, 2), "near")
far = E(trapezoid(1.8, 2, 3.3, 3.5), "far")
far_away = E(rtrapezoid(3.3, 3.5), "far away")

# neighbours expression
few = E(ltrapezoid(0.2, 0.3), "few")
rather = E(trapezoid(0.2, 0.3, 0.5, 0.6), "rather")
a_lot = E(rtrapezoid(0.5, 0.6), "a lot")

not_similiar = R(lab = far | far_away)
is_near = R(lab = near)
is_noised = R(neighbours = a_lot | rather)