
from scipy.optimize.nonlin import BroydenFirst, KrylovJacobian
from scipy.optimize.nonlin import InverseJacobian

jac = BroydenFirst()
kjac = KrylovJacobian(inner_M=InverseJacobian(jac))

