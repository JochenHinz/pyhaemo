#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Fabio Marcinno'
"""

import unittest
from splico.spl import NDSplineArray
import os
import sys
sys.path.append("/home/fabio/PhD/PyHaemo/examples/")
from data_handler import csv_converter

class TestMeshQualityCriteria(unittest.TestCase):

  def test_csv_converter(self):
      foldername = '/home/fabio/PyHaemo/test_haemo/data/centerlines_csv'
      centerlines_list, radii_list = [], []

      # generator expression
      filenames = (os.path.join(foldername, f) for f
                   in os.listdir(foldername) 
                   if f.endswith('.csv'))

      # Loop over each file
      for cspline, rspline in map(csv_converter, filenames):
          centerlines_list.append(cspline)
          radii_list.append(rspline)

      centerline_spline = NDSplineArray(centerlines_list)
      radius_spline = NDSplineArray(radii_list)


if __name__ == '__main__':
  unittest.main()
