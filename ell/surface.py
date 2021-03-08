#!/usr/bin/env python

import numpy as np
from scipy.integrate import quad
#import sys
#import warnings
#
#if not sys.warnoptions:
#    warnings.simplefilter("ignore")

__all__ = ["Surface"]


class Surface:
    def __init__(self, N=51):

        self.N        = int(N)

        if not self.N % 2:
            raise ValueError("grid size `N` needs to be an odd integer")


    def intensity(self, mu):
        raise NotImplementedError

    def get_brightness_weighted(self, I, y):
        """ computes the brightness-weighted average of y occulted by the planet"""
        assert len(y.shape) == 3 # check that input is 3d array

        return np.sum(I * y, axis=(1,2)) / np.sum(I, axis=(1,2))

    def get_convective_velocity(self,
            orbit=None,
            x=None, 
            c=None,
            oversample=1,
            texp=None):

        """ returns the convective contribution to the velocity """


        # function to integrate
        def _integrand(mu, i):
            """ function to integrate """
            intensity = self.get_intensity(mu)

            return intensity * mu**(i+1)

        c    = [0] + list(c)
        clen = len(c)

        integral = np.array([quad(_integrand, a=0, b=1, args=(i))[0]
                             for i in range(clen)])

        # check for nans
        assert ~np.any(np.isnan(integral)), "nan in integral"

        # don't quite understand why this is needed. MCMC could surely fit for
        # it?
#        if clen > 1:
        c0 = -np.sum(c[1:] * integral[1:]) / integral[0]
        c[0] = c0

        # oversample grid
        if texp is not None and oversample > 1:
            x = self._oversample_grid(x, texp, oversample)

        # compute the brightness weighted position on the disc
        mu_avg = self.get_mu(orbit, x)

        # compute the centre-to-limb convective blueshift velocity
        v = np.sum([c[i] * mu_avg**i for i in range(clen)], axis=0)

        # average the oversampled grid
        v_avg = np.array([np.mean(v[i:i+oversample])
                         for i in range(0, len(v), oversample)])

        return v_avg


    def get_rotational_velocity(self, 
            orbit=None,
            x=None,
            v_eq=None, 
            ell=None,
            i_star=90, 
            alpha=0,
            oversample=1, 
            texp=None
            ):

        ell    = np.deg2rad(ell)
        i_star = np.deg2rad(i_star)
        beta   = 0.5 * np.pi - i_star # rotation around orthogonal x axis
        n      = len(x)

        # oversample grid
        if texp is not None and oversample > 1:
            x = self._oversample_grid(x, texp, oversample)

        # get grid covering the planet at each epoch
        Xp, Yp, M = self._get_grid(orbit, x)

        # do transformations to the coordinate system
        Xn, Yn, Zn = orbit._transform_to_orthogonal(Xp, Yp, ell)
        _, Yrot, _ = orbit._rotate_around_x(Xn, Yn, Zn, i_star)

        V = self._v_rot(Xn, Yrot, v_eq, i_star, alpha)

        # get the intensity of the disc occulted by the planet
        MU    = orbit._xy_to_mu(Xp, Yp)
        I     = self.get_intensity(MU)
        I[~M] = 0

        # compute the brightness-weighted radial velocity due to rotation
        v = self.get_brightness_weighted(I, V)

        # average the oversampled grid
        return np.mean(
                v.reshape((n, oversample)), axis=1
                )

    def get_intensity(self, mu):
        return self.intensity(mu)

    def get_mu(self, orbit, x):

        # compute positions
        Xp, Yp, M = self._get_grid(orbit, x)
        MU        = orbit._xy_to_mu(Xp, Yp)
        I         = self.get_intensity(MU)

        print(np.count_nonzero(M))
        # only use cells inside planet/disc
        I[~M]  = 0
        MU[~M] = 0
    
        return self.get_brightness_weighted(I, MU)

    def get_latitudes(self, orbit, x, l, i_star):

        # compute positions and brightness
        Xp, Yp, M = self._get_grid(orbit, x)
        MU        = orbit._xy_to_mu(Xp, Yp)
        I         = self.get_intensity(MU)

        XYZ  = orbit._transform_to_orthogonal(Xp, Yp, l)
        Yrot = orbit._rotate_around_x(*XYZ, i_star)[1]

        I[~M]    = 0
        Yrot[~M] = 0

        yrot = self.get_brightness_weighted(I, Yrot)

        return np.rad2deg(np.arcsin(yrot))

    def _compute_grid(self, ror):

        grid = np.linspace(-ror, ror, self.N) 

        X, Y = np.meshgrid(grid, grid)

        return X, Y

    def _get_grid(self, orbit, x):

        # planet positions on the stellar disc
        xp, yp = orbit.get_planet_position(x)

        # compute a square grid covering the size of the planet
        X, Y   = self._compute_grid(orbit.ror)

        # create the grid for each planet position
        Xp = X + xp[:,None,None]
        Yp = Y + yp[:,None,None]

        # select points inside planet disc
        inside_planet = np.sqrt(X**2 + Y**2) <= orbit.ror

        # select points inside stellar disc
        inside_star   = np.sqrt(Xp**2 + Yp**2) <= 1

        # boolean mask where `True` points are behind the planet and on the
        # stellar disc
        M = inside_star & inside_planet

        return Xp, Yp, M

    def _oversample_grid(self, x, texp, oversample):
        return np.concatenate(
                [
                    np.linspace(_x - 0.5*texp, _x + 0.5*texp, oversample)
                               for _x in x
                               ]
                             )

    def _v_rot(self, x_orth, y_rot, v_eq, i_star, alpha):
        """ returns the stellar rotational velocity at a given latitude """
        v = (x_orth * v_eq * np.sin(i_star) * (1 - alpha * y_rot**2))

        # replace nans with zeros
        v[np.isnan(v)] = 0.0

        return v



