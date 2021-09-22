#!/usr/bin/env python

import numpy as np
#import sys
#import warnings
#
#if not sys.warnoptions:
#    warnings.simplefilter("ignore")

__all__ = ["Orbit"]

class Orbit:
    def __init__(self, roa=None, ror=None, i_pl=None, aor=None):

        if roa is not None and aor is None:
            self.roa = roa
            self.aor = 1 / self.roa
        elif roa is None and aor is not None:
            self.aor = aor
            self.roa = 1 / self.aor
        elif roa is not None and aor is not None:
            raise ValueError(
            "only one of `roa` or `aor` may be given, not both."
            )
        else:
            self.roa = None
            self.aor = None

        self.ror    = ror
        self.i_pl   = None if i_pl is None else np.deg2rad(i_pl)
        

    def __str__(self):
        return f"Orbit: a/R = {self.aor}, r/R = {self.ror}, i_pl = {np.rad2deg(self.i_pl)} deg"

    def get_planet_position(self, x):

        xp = np.sin(2 * np.pi * x) / self.roa
        yp = -np.cos(2 * np.pi * x) * np.cos(self.i_pl) / self.roa

        return xp, yp


    def get_planet_mu(self, x):
        xp, yp = self.get_planet_position(x)

        mu = self._xy_to_mu(xp, yp)

        return mu

    def _xy_to_mu(self, x, y):
        delta = np.sqrt(x**2 + y**2)

        mu = np.sqrt(1 - delta**2)

        return mu

#
    def _transform_to_orthogonal(self, x, y, l):

#        l = np.deg2rad(l)

        xn = x * np.cos(l) - y * np.sin(l)
        yn = x * np.sin(l) + y * np.cos(l)
        zn = np.sqrt(1 - xn**2 - yn**2)

        return xn, yn, zn

    def _rotate_around_x(self, x, y, z, i_star):

#        i_star = np.deg2rad(i_star)
        beta = 0.5*np.pi - i_star

        xrot = x
        yrot = z * np.sin(beta) + y * np.cos(beta)
        zrot = z * np.cos(beta) - y * np.sin(beta)
        
        return xrot, yrot, zrot


    def get_planet_position_orthogonal(self, x, l):

        xp, yp = self.get_planet_position(x)

        xn, yn, zn = self._transform_to_orthogonal(xp, yp, l)

        return xn, yn, zn

    def get_planet_position_rotated(self, x, l, i_star):

        xn, yn, zn       = self.get_planet_position_orthogonal(x, l)
        xrot, yrot, zrot = self._rotate_around_x(xn, yn, zn, i_star)

        return xrot, yrot, zrot

    def get_latitudes(self, x, l, i_star):
        """ returns the latitudes relative to the equator transited by the
            planet """

        yrot = self.get_planet_position_rotated(x, l, i_star)[1]

        return np.rad2deg(np.arcsin(yrot))


