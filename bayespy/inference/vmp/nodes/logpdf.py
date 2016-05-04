################################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np

from .expfamily import ExponentialFamily, useconstructor
from .stochastic import Distribution
from .node import Moments


class LogPDFDistribution(Distribution):
    pass


class LogPDF(ExponentialFamily):
    """
    General node with arbitrary probability density function
    """


    def __init__(self, logpdf, *parents, **kwargs):

        self._logpdf = logpdf

        super().__init__(logpdf,
                         *parents,
                         initialize=False,
                         **kwargs)


    @classmethod
    def _constructor(cls, logpdf, *parents, approximation=None, shape=None, samples=10, **kwargs):
        r"""
        Constructs distribution and moments objects.
        """

        if approximation is not None:
            raise NotImplementedError() #self._distribution = approximation._constructor

        dims = ( shape, )

        _distribution = LogPDFDistribution()

        _moments = np.nan

        _parent_moments = [Moments()] * len(parents)

        parent_plates = [_distribution.plates_from_parent(i, parent.plates)
                         for (i, parent) in enumerate(parents)]

        return (parents,
                kwargs,
                dims, 
                cls._total_plates(kwargs.get('plates'),
                                  *parent_plates),
                _distribution, 
                _moments, 
                _parent_moments)


    def _get_message_and_mask_to_parent(self, index):
        def logpdf_sampler(x):
            inputs = [self.parents[j].random() if j != index
                      else x
                      for j in range(len(self.parents))]
            return self._logpdf(self.random(), *inputs)
        mask = self._distribution.compute_weights_to_parent(index, self.mask) != 0
        return (logpdf_sampler, mask)


    def observe(self, x, *args, mask=True):
        """
        Fix moments, compute f and propagate mask.
        """

        # Compute fixed moments
        if not np.isnan(self._moments):
            u = self._moments.compute_fixed_moments(x, *args, mask=mask)
        else:
            u = (x,) + args

        # Check the dimensionality of the observations
        for (i,v) in enumerate(u):
            # This is what the dimensionality "should" be
            s = self.plates + self.dims[i]
            t = np.shape(v)
            if s != t:
                msg = "Dimensionality of the observations incorrect."
                msg += "\nShape of input: " + str(t)
                msg += "\nExpected shape: " + str(s)
                msg += "\nCheck plates."
                raise Exception(msg)

        # Set the moments
        self._set_moments(u, mask=mask)

        # Observed nodes should not be ignored
        self.observed = mask
        self._update_mask()

