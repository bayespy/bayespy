# (1) Construct the model
import bayespy as bp
mu = bp.nodes.Normal(0, 1e-3)
tau = bp.nodes.Gamma(1e-3, 1e-3)
y = bp.nodes.Normal(mu, tau, plates=(10,))
# (2) Observe data
import numpy as np
data = np.random.normal(0, 10, size=(10,))
y.observe(data)
# (3) Run inference
from bayespy.inference import VB
M = VB(mu, tau, y)
M.update(repeat=20)
# (4) Show posterior results
mu.show()
tau.show()

import bayespy.plotting as plt


