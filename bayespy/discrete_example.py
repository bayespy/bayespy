# This example could be simplified a little bit by using Bernoulli instead of
# Categorical, but Categorical makes it possible to use more categories than
# just TRUE and FALSE.

import numpy as np

from bayespy.nodes import Categorical, Mixture
from bayespy.inference import VB

# NOTE: Python's built-in booleans don't work nicely for indexing, thus define
# own variables:
FALSE = 0
TRUE = 1

def _or(p_false, p_true):
    """
    Build probability table for OR-operation of two parents

    p_false: Probability table to use if both are FALSE

    p_true: Probability table to use if one or both is TRUE
    """
    return np.take([p_false, p_true], [[FALSE, TRUE], [TRUE, TRUE]], axis=0)

asia = Categorical([0.5, 0.5])

tuberculosis = Mixture(asia, Categorical, [[0.99, 0.01], [0.8, 0.2]])

smoking = Categorical([0.5, 0.5])

lung = Mixture(smoking, Categorical, [[0.98, 0.02], [0.25, 0.75]])

bronchitis = Mixture(smoking, Categorical, [[0.97, 0.03], [0.08, 0.92]])

xray = Mixture(tuberculosis, Mixture, lung, Categorical,
               _or([0.96, 0.04], [0.115, 0.885]))

dyspnea = Mixture(bronchitis, Mixture, tuberculosis, Mixture, lung, Categorical,
                  [_or([0.6, 0.4], [0.18, 0.82]),
                   _or([0.11, 0.89], [0.04, 0.96])])

# Mark observations
tuberculosis.observe(TRUE)
smoking.observe(FALSE)
bronchitis.observe(TRUE) # not a "chance" observation as in the original example

# Run inference
Q = VB(dyspnea, xray, bronchitis, lung, smoking, tuberculosis, asia)
Q.update(repeat=100)

# Show results
print("P(asia):", asia.get_moments()[0][TRUE])
print("P(tuberculosis):", tuberculosis.get_moments()[0][TRUE])
print("P(smoking):", smoking.get_moments()[0][TRUE])
print("P(lung):", lung.get_moments()[0][TRUE])
print("P(bronchitis):", bronchitis.get_moments()[0][TRUE])
print("P(xray):", xray.get_moments()[0][TRUE])
print("P(dyspnea):", dyspnea.get_moments()[0][TRUE])
