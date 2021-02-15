#!/usr/bin/env python

import numpy as np

from surface import Surface

__all__ = ["UniformLimbDark", "LinLimbDark", "QuadLimbDark"]


class UniformLimbDark(Surface):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def intensity(self, mu):
        return np.ones_like(mu)

class LinLimbDark(Surface):
    
    def __init__(self, u, **kwargs):
        super().__init__(**kwargs)

        self.u = u

    def intensity(self, mu):
        return  1 - self.u * (1 - mu)

class QuadLimbDark(Surface):

    def __init__(self, u, **kwargs):
        super().__init__(**kwargs)

        self.u = u

    def intensity(self, mu):
        u, v = self.u
        return  1 - u * (1 - mu) - v * (1 - mu)**2


