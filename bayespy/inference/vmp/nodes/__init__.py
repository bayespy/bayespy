######################################################################
# Copyright (C) 2011,2012 Jaakko Luttinen
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
######################################################################

######################################################################
# This file is part of BayesPy.
#
# BayesPy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BayesPy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.
######################################################################

# Import some most commonly used nodes

from . import *

#from .node import Node
#from .variable import Variable
#from .constant import Constant
from .gaussian import Gaussian
from .wishart import Wishart
from .normal import Normal
from .gamma import Gamma
from .dirichlet import Dirichlet
from .categorical import Categorical
from .dot import Dot, MatrixDot
from .mixture import Mixture
from .gaussian_markov_chain import GaussianMarkovChain

## from . import variable

## import imp

## if 'Variable' in locals():
##     print('VARIABLE!')
##     #del Variable #imp.reload(Variable)
##     #del Dirichlet
##     #imp.reload(variable)

## import variable
    

## from .variable import Variable
## from .constant import Constant

## from .wishart import Wishart
## from .gaussian import Gaussian
## from .gamma import Gamma
## from .normal import Normal
## from .dirichlet import Dirichlet
## from .categorical import Categorical

## from .mixture import Mixture
## from .dot import Dot

## from . import variable
## from . import constant

## # BLAAH, they must in the correct order in order them to work.. :/
## from . import wishart
## from . import gaussian
## from . import gamma
## from . import normal
## from . import dirichlet
## from . import categorical

## from . import mixture
## from . import dot

## import constant

## # BLAAH, they must in the correct order in order them to work.. :/
## import wishart
## import gaussian
## import gamma
## import normal
## import dirichlet
## import categorical

## import mixture
## import dot


## import imp
## imp.reload(variable)
## imp.reload(constant)
## imp.reload(wishart)
## imp.reload(gaussian)
## imp.reload(gamma)
## imp.reload(normal)
## imp.reload(dirichlet)
## imp.reload(categorical)
## imp.reload(mixture)
## imp.reload(dot)



## from .Gaussian import Gaussian
## from .Wishart import Wishart
## from .Normal import Normal
## from .Gamma import Gamma
## from .Dirichlet import Dirichlet
## from .Categorical import Categorical

## from .Mixture import Mixture
## from .Dot import Dot
