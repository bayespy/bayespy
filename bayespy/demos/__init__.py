######################################################################
# Copyright (C) 2013 Jaakko Luttinen
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

from . import demo_categorical
from . import demo_gaussian_markov_chain
from . import demo_gaussian_mixture
#from . import demo_gp
#from . import demo_gp_sparse
#from . import demo_gpfa
from . import demo_kalman_filter
from . import demo_lssm
#from . import demo_lssm_drift
from . import demo_pca
#from . import demo_rotation
from . import demo_saving

def run_all():
    demos = [
        demo_categorical,
        demo_gaussian_markov_chain,
        demo_gaussian_mixture,
    #demo_gp,
    #demo_gp_sparse,
    #demo_gpfa,
        demo_kalman_filter,
        demo_lssm,
    #demo_lssm_drift,
        demo_pca,
        demo_rotation,
        demo_saving
        ]

    for demo in demos:
        try:
            demo.run()
        except:
            print("DEMO FAILURE")
            pass

