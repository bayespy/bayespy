import bayespy.plot as bpplt

def setup():
    for i in bpplt.pyplot.get_fignums():
        fig = bpplt.pyplot.figure(i)
        fig.clear()
