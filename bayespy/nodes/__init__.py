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

from .node import Node

from .variables.variable import Variable
from .variables.constant import Constant
from .variables.gaussian import Gaussian
from .variables.wishart import Wishart
from .variables.normal import Normal
from .variables.gamma import Gamma, GammaToDiagonalWishart
from .variables.dirichlet import Dirichlet
from .variables.categorical import Categorical
from .variables.dot import Dot
from .variables.mixture import Mixture

#print('JELOU')

## #import nodes.node as node
## #from nodes import node
## #from . import node
## #from .node import Node
## from . import *

## from .variables import (variable,
##                         constant,
##                         gaussian,
##                         wishart,
##                         normal,
##                         gamma,
##                         dirichlet,
##                         categorical,
##                         dot,
##                         mixture)

## ## import imp

## ## imp.reload(node)

## ## imp.reload(variable)
## ## imp.reload(constant)
## ## imp.reload(gaussian)
## ## imp.reload(wishart)
## ## imp.reload(normal)
## ## imp.reload(gamma)
## ## imp.reload(dirichlet)
## ## imp.reload(categorical)
## ## imp.reload(dot)
## ## imp.reload(mixture)

## #Node = node.Node

## Variable = variable.Variable
## Constant = constant.Constant
## Gaussian = gaussian.Gaussian
## Wishart = wishart.Wishart
## Normal = normal.Normal
## Gamma = gamma.Gamma
## GammaToDiagonalWishart = gamma.GammaToDiagonalWishart
## Dirichlet = dirichlet.Dirichlet
## Categorical = categorical.Categorical
## Dot = dot.Dot
## Mixture = mixture.Mixture

## ## from .variable import Variable
## ## from .constant import Constant
## ## from .gaussian import Gaussian
## ## from .wishart import Wishart
## ## from .normal import Normal
## ## from .gamma import Gamma
## ## from .dirichlet import Dirichlet
## ## from .categorical import Categorical
## ## from .dot import Dirichlet
## ## from .mixture import Categorical


