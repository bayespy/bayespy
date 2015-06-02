################################################################################
# Copyright (C) 2011-2013 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


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

