######################################################################
# Copyright (C) 2011-2013 Jaakko Luttinen
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
######################################################################

######################################################################
# This file is part of BayesPy.
#
# BayesPy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# BayesPy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.
######################################################################

import numpy as np

from bayespy.utils import random
from bayespy import nodes

from bayespy.inference.vmp.vmp import VB

def run(M=30, D=5):

    # Generate data
    y = np.random.randint(D, size=(M,))

    # Construct model
    p = nodes.Dirichlet(1*np.ones(D),
                        name='p')
    z = nodes.Categorical(p, 
                          plates=(M,), 
                          name='z')

    # Observe the data with randomly missing values
    mask = random.mask(M, p=0.5)
    z.observe(y, mask=mask)

    # Run VB-EM
    Q = VB(p, z)
    Q.update()

    # Show results
    z.show()
    p.show()

if __name__ == '__main__':
    run()

