
#!/usr/bin/env python

import numpy as np
import pytest

from ell.limbdark import *
from ell.orbit import Orbit


def test_rotational_velocity():

    roa  = 0.114
    ror  = 0.103
    i_pl = 90

    orbit = Orbit(roa=roa, ror=ror, i_pl=i_pl)

    edge = 1 + orbit.ror - 5e-4
    x = np.array([-edge, -0.5, 0, 0.5, edge])
    t = np.arcsin(x * orbit.roa) / (2 * np.pi)

    # check v_eq at limb is 3 km/s for vsini = 3 km/s and i_pl = 90 deg
    # test uniform brightness stellar disc
    v = UniformLimbDark(N=501).get_rotational_velocity(
            orbit=orbit,
            x=t,
            v_eq=3,
            ell=0
            )
    assert v.shape == t.shape
    assert np.allclose(v, [-3, -1.5, 0, 1.5, 3], atol=1e-3)

    # test linear limb darkening
    v = LinLimbDark([0.3], N=501).get_rotational_velocity(
            orbit=orbit,
            x=t,
            v_eq=3,
            ell=0
            )
    assert np.allclose(v, [-3, -1.5, 0, 1.5, 3], atol=1e-2)

    # test quadratic limb darkening
    v = QuadLimbDark([0.4, 0.3], N=501).get_rotational_velocity(
            orbit=orbit,
            x=t,
            v_eq=3,
            ell=0
            )
    assert np.allclose(v, [-3, -1.5, 0, 1.5, 3], atol=1e-2)

def test_convetive_velocity():
    # HD189733b test case


    pass


