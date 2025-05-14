"""
This module provides functions to handle and convert data into spline representations.
"""

import warnings
from typing import Literal
from splico.spl import UnivariateKnotVector
from splico.util import np
from splico.util import clparam as chord_length

FORMAT = 'csv', 'swc'


def csv_converter(filename: str):
    """
    Convert a CSV file containing centerline and radius data into spline
    representations.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the centerline spline and radius spline.
    """
    process_input(filename, 'csv')

    data = np.genfromtxt(filename, delimiter=',', skip_header=1)

    centerline, radius = data_extraction(data)

    centerline_spline, radius_spline = to_spline(centerline, radius)

    return centerline_spline, radius_spline


def process_input(filename: str, format: Literal[FORMAT]):
    """
    Processes the input file based on the specified format.

    Args:
        filename (str): The path to the input file. It will be
        stripped of leading and trailing whitespace.
        format (Literal[FORMAT]): The format of the input file.
        Must be one of the predefined formats in FORMAT.

    Raises:
        NotImplementedError: If the specified format is not supported.
    """
    filename = str(filename).strip()

    if format not in FORMAT:
        raise NotImplementedError(f'Not implemented for format: {format}')


def data_extraction(data):
    """
    Extracts centerline and radius data from the input array.

    Args:
        data (numpy.ndarray): The input data array with centerline
                              and radius information.

    Returns:
        tuple: A tuple containing the centerline and radius arrays.
    """
    centerline = data[:, :3]
    radius = data[:, -1]

    if len(centerline) != len(radius):
        warnings.warn("The number of points is not equal to radii")

    return centerline, radius


def to_spline(centerline, radius, n_points=100):
    """
    Converts a single centerline and radius into spline representations.

    Args:
        centerline (numpy.ndarray): The centerline data points.
        radius (numpy.ndarray): The radius data points.
        n_points (int): Number of points for the spline representation.

    Returns:
        tuple: A tuple containing the centerline spline and radius spline.
    """
    kv_center = UnivariateKnotVector(np.linspace(0, 1, n_points)).to_tensor()
    #kv_center = TensorKnotVector([kv_center])

    # Compute the chord length parameter values of the centerline points
    xi_center = chord_length(centerline)
    xi_radius = chord_length(radius)

    # Fit a univariate spline to the data
    centerline_spline = kv_center.fit([xi_center], centerline)
    radius_spline = kv_center.fit([xi_radius], radius)

    return centerline_spline, radius_spline
