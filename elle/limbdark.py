#!/usr/bin/env python

__all__ = ["QuadLimbDark"]


class LimbDark:
    npar = None
    name = None


class QuadLimbDark(LimbDark):
    npar = 2
    name = 'quad'

    def intensity(mu, pars):
        u, v = pars
        return  1 - u * (1 - mu) - v * (1 - mu)**2


