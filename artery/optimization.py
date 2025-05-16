import artery
from pyhaemo.util import np, _, _round_array, clparam
from pyhaemo.types import Immutable, Int
from pyhaemo.spl import NDSpline, UnivariateKnotVector
from pyhaemo.geo.interp import linear_interpolation
from pyhaemo.log import logger as log

from datahandler import csv_converter, swc_converter
from aux import centerline_length
from plot import plot_frames, plot_artery

from itertools import chain, combinations, count
from typing import Tuple, Sequence
import os

from numpy.typing import NDArray
from scipy.spatial import cKDTree

from artery import Artery

import jax
import jax.numpy as jnp



#IDEA: the optimization should works on the deltanormal, where deltanormal is 
# the rotation we need to add to the normal vector to get the new normal vector.
# In this way we do not touch the self.normals_bif, but we will obtain
# new_normals = self.normals_bif + deltanormal

class Optimizer(Immutable):

    def __init__(self, artery: Artery, normals_bif: NDArray):
        self.artery = artery
        self.normals_bif = normals_bif
        len(self.artery.nontrivial_bifs) == len(self.normals_bif)

    def cost_function(self):
        pass

    def optimize(self):
        pass
