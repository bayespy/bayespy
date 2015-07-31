

from bayespy import nodes
from bayespy.inference import VB


def run():

    a = nodes.GammaShape(name='a')
    b = nodes.Gamma(1e-5, 1e-5, name='b')

    tau = nodes.Gamma(a, b, plates=(1000,), name='tau')
    tau.observe(nodes.Gamma(10, 20, plates=(1000,)).random())

    Q = VB(tau, a, b)

    Q.update(repeat=1000)

    print("True gamma parameters:", 10.0, 20.0)
    print("Estimated parameters from 1000 samples:", a.u[0], b.u[0])


if __name__ == "__main__":

    
    run()
