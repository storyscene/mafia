import numpy as np
from scipy.stats import norm

class Variable():
    def __init__(self):
        pass


class Normal(Variable):
    def __init__(self, mu, sig):
        super().__init__()
        self.mu = mu
        self.sig = sig
        assert self.sig > 0, \
            f'received sig = {sig:.4f}, must have sig > 0'

    def pdf(self, x):
        return norm.pdf(x, loc=self.mu, scale=self.sig)

    def cdf(self, x):
        return norm.cdf(x, loc=self.mu, scale=self.sig)
    
    def sample(self, size=None):
        return np.random.normal(loc=self.mu, scale=self.sig, size=size)

    def __repr__(self):
        return f'Normal({self.mu}, {self.sig})'

class Bernoulli(Variable):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def pdf(self, x):
        return x * self.p + (1-x) * (1-self.p)
    
    def sample(self, size=None):
        return (np.random.random(size=size) < self.p).astype(int)
    
class Uniform(Variable):
    def __init__(self, rmin, rmax):
        super().__init__()
        self.rmin = rmin
        self.rmax = rmax
        self.range = rmax - rmin
    
    def pdf(self, x):
        return np.ones(shape=x.shape) / (self.rmax - self.rmin)
    
    def sample(self, x):
        return (np.random.random() * self.range) + self.rmin