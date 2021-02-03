#!/usr/bin/env python

import numpy as np

__all__ = ["UniformLimbDark", "QuadraticLimbDark"]


class LimbDark:

    def intensity(mu):
        raise NotImplementedError

    
class QuadraticLimbDark(LimbDark):

    def intensity(mu, pars):
        u, v = pars
        return  1 - u * (1 - mu) - v * (1 - mu)**2

class UniformLimbDark(LimbDark):

    def intensity(mu):
        return np.ones_like(mu)
