
from nodes.node import Node

from .exponential_family import (variable,
                                 constant,
                                 gaussian,
                                 wishart,
                                 normal,
                                 gamma,
                                 dirichlet,
                                 categorical,
                                 dot,
                                 mixture)

import imp

imp.reload(variable)
imp.reload(constant)
imp.reload(gaussian)
imp.reload(wishart)
imp.reload(normal)
imp.reload(gamma)
imp.reload(dirichlet)
imp.reload(categorical)
imp.reload(dot)
imp.reload(mixture)


Variable = variable.Variable
Constant = constant.Constant
Gaussian = gaussian.Gaussian
Wishart = wishart.Wishart
Normal = normal.Normal
Gamma = gamma.Gamma
Dirichlet = dirichlet.Dirichlet
Categorical = categorical.Categorical
Dot = dot.Dot
Mixture = mixture.Mixture

## from .variable import Variable
## from .constant import Constant
## from .gaussian import Gaussian
## from .wishart import Wishart
## from .normal import Normal
## from .gamma import Gamma
## from .dirichlet import Dirichlet
## from .categorical import Categorical
## from .dot import Dirichlet
## from .mixture import Categorical


