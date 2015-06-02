################################################################################
# Copyright (C) 2015 Hannu Hartikainen
#
# This file is licensed under the MIT License.
################################################################################


import bayespy.plot as bpplt

def setup():
    for i in bpplt.pyplot.get_fignums():
        fig = bpplt.pyplot.figure(i)
        fig.clear()
