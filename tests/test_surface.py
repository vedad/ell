
#!/usr/bin/env python

import numpy as np
import pytest
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from elle.limbdark import *
from elle.orbit import Orbit

def test_input():
    pass
    

    # 

def test_uniform():

    ld = UniformLimbDark()
    assert ld.N == 51

    ld = UniformLimbDark(N=31)
    assert ld.N == 31


    roa  = 0.114
    ror  = 0.103
    i_pl = 90
    
    orbit = Orbit(roa=roa, ror=ror, i_pl=i_pl)

    edge = 1 + orbit.ror - 5e-4
    x = np.array([-edge, -0.5, 0, 0.5, edge])
    t = np.arcsin(x * orbit.roa) / (2 * np.pi)

    # check v_eq at limb is 3 km/s for vsini = 3 km/s and i_pl = 90 deg
    v = UniformLimbDark(N=501).get_rotational_velocity(
            orbit=orbit,
            x=t,
            v_eq=3,
            ell=0
            )
    assert v.shape == t.shape
    assert np.allclose(v, [-3, -1.5, 0, 1.5, 3], atol=1e-3)


    
def test_linear():
    ld = LinLimbDark([0.3])
    assert ld.N == 51
    assert ld.u == [0.3]

    ld = LinLimbDark(0.3, N=31)
    assert ld.N == 31
    assert ld.u == [0.3]

    roa  = 0.114
    ror  = 0.103
    i_pl = 90
    
    orbit = Orbit(roa=roa, ror=ror, i_pl=i_pl)

    edge = 1 + orbit.ror - 5e-4
    x = np.array([-edge, -0.5, 0, 0.5, edge])
    t = np.arcsin(x * orbit.roa) / (2 * np.pi)

    # check v_eq at limb is 3 km/s for vsini = 3 km/s and i_pl = 90 deg
    v = LinLimbDark([0.3], N=501).get_rotational_velocity(
            orbit=orbit,
            x=t,
            v_eq=3,
            ell=0
            )
    assert v.shape == t.shape
    assert np.allclose(v, [-3, -1.5, 0, 1.5, 3], atol=1e-2)


def test_quadratic():
    ld = QuadLimbDark([0.4, 0.3])
    assert ld.N == 51
    assert ld.u == [0.4, 0.3]


    ld = QuadLimbDark([0.4, 0.3], N=31)
    assert ld.N == 31

    roa  = 0.114
    ror  = 0.103
    i_pl = 90
    
    orbit = Orbit(roa=roa, ror=ror, i_pl=i_pl)

    edge = 1 + orbit.ror - 5e-4
    x = np.array([-edge, -0.5, 0, 0.5, edge])
    t = np.arcsin(x * orbit.roa) / (2 * np.pi)

    # check v_eq at limb is 3 km/s for vsini = 3 km/s and i_pl = 90 deg
    v = QuadLimbDark([0.4, 0.3], N=501).get_rotational_velocity(
            orbit=orbit,
            x=t,
            v_eq=3,
            ell=0
            )
    assert v.shape == t.shape
    assert np.allclose(v, [-3, -1.5, 0, 1.5, 3], atol=1e-2)


def test_rotational_velocity():
    pass

def test_convetive_velocity():
    pass


