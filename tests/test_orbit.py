#!/usr/bin/env python

import numpy as np
import pytest

from ell.orbit import Orbit

def test_input():

    # roughly 3-day Jupiter planet orbiting the Sun
    roa  = 0.114
    aor  = 8.772
    ror  = 0.103
    i_pl = 90

    # test no args
    orbit = Orbit()
    assert orbit.roa is None
    assert orbit.aor is None
    assert orbit.i_pl is None
    assert orbit.ror is None

    orbit = Orbit(roa=roa, ror=ror, i_pl=i_pl)

    # test i_p is transformed to rad
    assert np.isclose(orbit.i_pl, 1.5708)

    # test ror is an attribute
    assert np.isclose(orbit.ror, 0.103)

    # test roa, incl, ror
    assert np.isclose(orbit.roa, 0.114)
    assert np.isclose(orbit.aor, 8.772)

    # test aor, incl, ror
    orbit = Orbit(aor=aor, ror=ror, i_pl=i_pl)
    assert np.isclose(orbit.aor, 8.772)
    assert np.isclose(orbit.roa, 0.114)

def test_planet_position():

    roa  = 0.114
    ror  = 0.103
    i_pl = 90
    
    orbit = Orbit(roa=roa, ror=ror, i_pl=i_pl)

    #  phase where planet centre is at limb (x=-1)
    t  = np.arcsin(-roa) / (2*np.pi)
    xy = orbit.get_planet_position(t)

    # check output is separate x and y coordinates
    assert len(xy) == 2

    # check position at limb
    assert np.allclose(xy, [-1, 0])

    # check position at centre
    xy = orbit.get_planet_position(0)
    assert np.allclose(xy, [0, 0])

    # transit phase duration of a 3-day Jupiter planet orbiting the Sun is
    # roughly 0.04
    t = np.zeros(5)
    x, y = orbit.get_planet_position(t)

    # check input and output shapes match
    assert x.shape == y.shape == t.shape


def test_planet_mu():

    roa  = 0.114
    ror  = 0.103
    i_pl = 90
    
    orbit = Orbit(roa=roa, ror=ror, i_pl=i_pl)

    # phase where planet centre is at limb (x=-1)
    t = np.arcsin(-roa) / (2*np.pi) 
    
    # test values at limb
    mu = orbit.get_planet_mu(t)
    assert np.isclose(mu, 0)

    # test values at centre
    mu = orbit.get_planet_mu(0)
    assert np.isclose(mu, 1)

    # check input and output shapes match
    t = np.zeros(5)
    mu = orbit.get_planet_mu(t)
    assert mu.shape == t.shape


def test_transformations():

    roa  = 0.114
    ror  = 0.103
    i_pl = 90
    
    orbit = Orbit(roa=roa, ror=ror, i_pl=i_pl)


    ## test orthogonal transformations
    xyz = orbit._transform_to_orthogonal(0.5, 0.5, 45)
    assert len(xyz) == 3
    assert np.allclose(xyz, [0, np.sqrt(2)/2, np.sqrt(2)/2])

    # test transformations with stellar inclination, i_star = np.pi/4
    xyz = orbit._rotate_around_x(
            0, 
            np.sqrt(2)/2, 
            np.sqrt(2)/2, 
            45
            )

    assert len(xyz) == 3
    assert np.allclose(xyz, [0.0, 1.0, 0.0]) 

    # test with phase array input
    t       = np.zeros(5)
    xyz     = orbit.get_planet_position_orthogonal(t, 0)
    x, y, z = xyz

    assert len(xyz) == 3
    assert x.shape == y.shape == z.shape == t.shape


def test_latitudes():
    roa  = 0.114
    ror  = 0.103
    i_pl = 90
    
    orbit = Orbit(roa=roa, ror=ror, i_pl=i_pl)

    #  phase where planet centre is at limb (x=-1)
    t = np.round(np.arcsin(-roa) / (2*np.pi), 10)

    # lambda = 0
    lat = orbit.get_latitudes(t, 0, 90)
    assert np.isclose(lat, 0)

    # lambda = 45 deg
    lat = orbit.get_latitudes(t, 45, 90)
    assert np.isclose(lat, -45)

    # lambda = 0, i_star = 0
    lat = orbit.get_latitudes(0, 0, 0)
    assert np.isclose(lat, 90)



    
