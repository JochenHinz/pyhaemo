#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Fabio Marcinno'
"""

from splico.mesh.qual import aspect_ratio
from splico.mesh import rectilinear
from test_mesh import unit_disc_triangulation
import unittest


class TestMeshQualityCriteria(unittest.TestCase):

  def test_aspect_ratio(self):
    mesh_unstruct = unit_disc_triangulation()
    mesh_struct = rectilinear((17, 25, 13))

    stats_unstruct = aspect_ratio(mesh_unstruct)
    stats_struct = aspect_ratio(mesh_struct)

    self.assertTrue(stats_unstruct[0] >= stats_unstruct[2])
    self.assertTrue(stats_unstruct[0] <= stats_unstruct[1])

    self.assertTrue(stats_struct[0] >= stats_struct[2])
    self.assertTrue(stats_struct[0] <= stats_struct[1])


if __name__ == '__main__':
  unittest.main()
