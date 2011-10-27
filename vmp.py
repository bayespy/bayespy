
from numpy import *
from scipy.special import digamma
from scipy.linalg.decomp_cholesky import cho_factor, cho_solve

def cho_logdet(L):
    return sum(log(diag(L[0])))

def cho_inv(L):
    return cho_solve(L, identity(L[0].shape[0]), overwrite_b=True)

def cov_logdet(C):
    return cho_logdet(cho_factor(C)[0])

def multivariate_digamma(a, d):
    y = 0
    for i in range(d):
        y += digamma(a + 0.5*(1-i))
    return y

## def sum_axis(A, s):
##     return sum(A, axis=tuple(where(array(s)==1)[0]), keepdims=True)

##     dim_A = A.shape
##     a = A
##     for j in range(len(dim_A)):
##         if j >= len(s) or s[j] == 1:
##             a = sum(a, axis=j, keepdims=True)
##     #a.reshape(s)
##     return a

class Node:

    # Proposed functions:
    def logpdf_integrated(self):
        # The log pdf when this node is integrated out (useful for
        # type-2 ML or predictive densities)
        return

    def random(self):
        # Draw a random variable from the node
        return

    def gibbs_sampling(self):
        # Hmm.. For Gibbs and for generating samples from the model?
        return

    def __init__(self, *args):
        # Parents
        self.parents = args
        # Inform parent nodes
        for parent in self.parents:
            parent.add_child(self)
        # Children
        self.children = list()
        # Natural parameters
        self.phi = list()
        # Moments
        self.u = list()
        # Not observed
        self.fixed = False

    def add_child(self, child):
        self.children.append(child)

    def update(self):
        if not self.fixed:
            # Messages from parents
            u_parents = list()
            for parent in self.parents:
                u_parents.append(parent.message_to_child())
            # Update natural parameters
            self.update_phi(u_parents)
            # Messages from children (just add to phi)
            for child in self.children:
                m = child.message_to_parent(self)
                for i in range(len(self.phi)):
                    #print("Update")
                    #print(self.__class__)
                    #print(self.phi[i])
                    #print(m[i])
                    self.phi[i] += m[i]
                    print("Message to phi")
                    print(m[i])
                    print(self.phi[i])
            # Update moments
            self.update_moments()

    def message_to_child(self):
        return self.u

    def message_to_parent(self, to):
        index = -1
        u_parents = list()
        for i, parent in enumerate(self.parents):
            u_parents.append(parent.message_to_child())
            if to == parent:
                index = i
        if i >= 0:
            m = self.message(index, u_parents)
            # Sum over singleton dimensions
            for i in range(len(m)):
                if isscalar(m[i]):
                    #print("Scalar message!")
                    #print(m[i])
                    m[i] = m[i]
                else:
                    shape_u = u_parents[index][i].shape
                    # Sum the dimensions of the message matrix to
                    # match the dimensionality of the parents natural
                    # parameterization
                    s = [j for j in range(len(m[i].shape)) \
                         if j >= len(shape_u) or shape_u[j] == 1]
                    m[i] = sum(m[i], axis=tuple(s), keepdims=True)
                    #print("Message")
                    #print(self.__class__)
                    #print(shape_u)
                    #print(s)
                    #print(m[i].shape)
                    m[i].shape = shape_u
                    #print(m[i].shape)
                    #print("Message ends")
            return m
        else:
            # Unknown parent
            raise Exception("Unknown parent requesting a message")

    def update_phi(self, u_parents):
        raise Exception("Not implemented.")
        pass

    def update_moments(self):
        raise Exception("Not implemented.")
        pass

    def message(self, index, u_parents):
        raise Exception("Not implemented.")
        pass

    def fix(self, u):
        self.u = u
        self.fixed = True

class NodeConstant(Node):
    def __init__(self, u):
        Node.__init__(self)
        self.fix(u)

class NodeGamma(Node):

    # Gamma(a, b)

    def __init__(self, a, b):
        # Check for constant a
        if isscalar(a):
            a = array([a])
        if isinstance(a, list):
            a = array(a)
        if isinstance(a, ndarray):
            a = NodeConstant([a])

        # Check for constant b
        if isscalar(b):
            b = array([b])
        if isinstance(b, list):
            b = array(b)
        if isinstance(b, ndarray):
            b = NodeConstant([b, log(b)])

        # Construct
        Node.__init__(self, a, b)

    def update_phi(self, u_parents):
        self.phi = [-u_parents[1][0],
                    u_parents[0][0].copy()]

    def update_moments(self):
        self.u = [self.phi[1] / (-self.phi[0]),
                  digamma(self.phi[1]) - log(-self.phi[0])]

    def message(self, index, u_parents):
        if index == 0:
            raise Exception("No analytic solution exists")
        elif index == 1:
            return [-self.u[0],
                    self.u_parents[0][0]]

    def show(self):
        a = self.phi[1]
        b = -self.phi[0]
        print("Gamma(" + str(a) + ", " + str(b) + ")")

class NodeNormal(Node):

    # Normal(mu, 1/tau)

    def __init__(self, mu, tau):
        # Check for constant mu
        if isscalar(mu):
            mu = array([mu])
        if isinstance(mu, list):
            mu = array(mu)
        if isinstance(mu, ndarray):
            mu = NodeConstant([mu, mu**2])

        # Check for constant tau
        if isscalar(tau):
            tau = array([tau])
        if isinstance(tau, list):
            tau = array(tau)
        if isinstance(tau, ndarray):
            tau = NodeConstant([tau, log(tau)])

        # Construct
        Node.__init__(self, mu, tau)

    def update_phi(self, u_parents):
        self.phi = [u_parents[1][0] * u_parents[0][0],
                    -u_parents[1][0] / 2]

    def update_moments(self):
        mu = -self.phi[0] / (2*self.phi[1])
        self.u = [mu,
                  mu**2 - 1 / (2*self.phi[1])]

    def message(self, index, u_parents):
        if index == 0:
            return [u_parents[1][0] * self.u[0],
                    -0.5 * u_parents[1][0]]
        elif index == 1:
            return [-0.5 * (self.u[1] - 2*self.u[0]*u_parents[0][0] + u_parents[0][1]),
                    0.5]

    def fix(self, x):
        Node.fix(self, [x, x**2])

    def show(self):
        mu = self.u[0]
        s2 = self.u[1] - mu**2
        print("Gaussian(" + str(mu) + ", " + str(s2) + ")")


class NodeWishart(Node):

    # Wishart(n, inv(V))

    def __init__(self, n, V):

        # Check for constant n
        if isscalar(n):
            n = array([n])
            #n = NodeConstant([n])
        if isinstance(n, list):
            n = array(n)
        if isinstance(n, ndarray):            
            n = NodeConstant(n)

        # Check for constant V
        if isscalar(V):
            V = array([V])
        if isinstance(V, list):
            V = array(V)
        if isinstance(V, ndarray):
            V = NodeConstant([V, cov_logdet(V)])

        Node.__init__(self, n, V)
        
    def update_phi(self, u_parents):
        self.phi = [-0.5 * u_parents[1][0],
                    0.5 * u_parents[0][0]]

    def update_moments(self):
        L = cho_factor(-self.phi[0])
        k = L[0].shape[0]
        self.u = [self.phi[1] * cho_inv(L),
                  -cho_logdet(L) + multivariate_digamma(self.phi[1], k)]

    def message(self, index, u_parents):
        if index == 0:
            raise Exception("No analytic solution exists")
        elif index == 1:
            # x_mu = dot(self.u[0], u_parents[0][0].T)
            return [-0.5 * self.u[0],
                    0.5 * self.u_parents[0][0]]

    def show(self):
        print("Wishart(n, A)")
        print("  n =")
        print(2*self.phi[1])
        print("  A =")
        print(0.5 * self.u[0] / self.phi[1])

class NodeGaussian(Node):

    # Gaussian(mu, inv(Lambda))

    def __init__(self, mu, Lambda):
        # mu must have size:     N x A/1 x B/1 x C/1 x ...
        # Lambda must have size: N x N x A/1 x B/1 x C/1 x ...

        # Check for constant mu
        if isscalar(mu):
            mu = array([mu])
        if isinstance(mu, list):
            mu = array(mu)
        if isinstance(mu, ndarray):
            mu = NodeConstant([mu, outer(mu, mu)])

        # Check for constant Lambda
        if isscalar(Lambda):
            Lambda = array([Lambda])
        if isinstance(Lambda, list):
            Lambda = array(Lambda)
        if isinstance(Lambda, ndarray):
            Lambda = NodeConstant([Lambda, cov_logdet(Lambda)])

        # Construct
        Node.__init__(self, mu, Lambda)

    def update_phi(self, u_parents):
        self.phi = [dot(u_parents[1][0], u_parents[0][0]),
                    -0.5 * u_parents[1][0]]
        #print self.phi

    def update_moments(self):
        L = cho_factor(-self.phi[1])
        mu = 0.5 * cho_solve(L, self.phi[0])
        Cov = 0.5 * cho_solve(L, identity(L[0].shape[0]))
        mumu = outer(mu, mu)
        self.u = [mu,
                  mumu + Cov]

    def message(self, index, u_parents):
        if index == 0:
            return [dot(u_parents[1][0], self.u[0]),
                    -0.5 * u_parents[1][0]]
        elif index == 1:
            xmu = outer(self.u[0], u_parents[0][0])
            return [-0.5 * (self.u[1] - xmu - xmu.T + u_parents[0][1]),
                    0.5]

    def show(self):
        mu = self.u[0]
        Cov = self.u[1] - outer(mu, mu)
        print("Gaussian(mu, Cov)")
        print("  mu = ")
        print(mu)
        print("  Cov = ")
        print(str(Cov))

    def fix(self, x):
        Node.fix(self, [x, outer(x,x)])

## # TODO: How to do constant nodes?
## # x = NodeGaussian(array([0, 0]), array([[1, 0.5], [0.5, 1]])
        
## # TODO: How to do observations?
## # y.observe(data)

# TODO: How to do missing values?

# TODO: Compute lower bound!


def test_normal():
    
    # mu
    mu = NodeNormal(0.0, 10**-5)
    print("Prior for mu:")
    mu.update()
    mu.show()

    # Lambda
    tau = NodeGamma(10**-5, 10**-5)
    print("Prior for tau:")
    tau.update()
    tau.show()

    # y (generate data)
    y = NodeNormal(mu, tau)
    y.fix(random.normal(0,1, (10,10)))
    ## for i in range(100):
    ##     y = NodeNormal(mu, tau)
    ##     y.fix(random.normal(0,1, 1))

    # Inference
    for i in range(50):
        mu.update()
        tau.update()

    print("Posterior for mu:")
    mu.show()
    print("Posterior for tau:")
    tau.show()
    
    return
    
def test_multivariate():    

    D = 3

    # mu
    mu = NodeGaussian(zeros(D), identity(D))
    print("Prior for mu:")
    mu.update()
    mu.show()

    # Lambda
    Lambda = NodeWishart(D, (10**-5) * identity(D))
    print("Prior for Lambda:")
    Lambda.update()
    Lambda.show()

    # y (generate data)
    for i in range(100):
        y = NodeGaussian(mu, Lambda)
        v = random.normal(0,1, D)
        y.fix(v)

    # Inference
    for i in range(50):
        mu.update()
        Lambda.update()

    print("Posterior for mu:")
    mu.show()
    print("Posterior for Lambda:")
    Lambda.show()
    

if __name__ == '__main__':
    test_normal()
