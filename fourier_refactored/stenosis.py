import numpy as np
from pyhaemo.spl import UnivariateKnotVector
from pyhaemo.util import clparam
from typing import Literal

import sys
from paths import path_to_directory
sys.path.append(str(path_to_directory.parent / 'partery'))
from fit import fit_constrained


def find_position(position: Literal['middle up', 'middle bottom'],
                  n_points: int):

    if position == "middle up":
        start_point = int(n_points/2 - n_points/3)
        end_point = int(n_points/2)

    elif position == "middle bottom":
        start_point = int(n_points/2)
        end_point = int(n_points/2 + n_points/3)

    else:
        raise AssertionError("Unexpected keyword")

    return start_point, end_point


def create_stenosis(radii,
                    position,
                    severity,
                    n_points):

    radii = radii.transpose(1, 0)
    start, end = find_position(position, n_points)
    mean = int(start/2 + end/2)
    kv = UnivariateKnotVector(np.linspace(0, 1, int(n_points)), degree=3).to_tensor()

    stenotic_radii = []
    for myradius in radii:
        stenotic_radius = np.copy(myradius)

        if severity == "medium":
            stenotic_radius[mean] = (stenotic_radius[start] + stenotic_radius[end])/2.5
        elif severity == "high":
            stenotic_radius[mean] = (stenotic_radius[start] + stenotic_radius[end])/4
        else:
            raise AssertionError("Unexpected keyword")

        rads= np.array(([ stenotic_radius[start],
                          stenotic_radius[mean],
                          stenotic_radius[end] ]))

        xi = clparam(rads)

        X_rad = fit_constrained(kv=kv,
                                absc=xi,
                                data=rads,
                                lam0=1e-5,
                                lam1=0)

        stenotic_radius[start:end] = X_rad(np.linspace(0, 1, end-start))
        stenotic_radii.append(stenotic_radius)

    stenotic_radii = np.array(stenotic_radii)

    return stenotic_radii
