from pyhaemo.util import np, clparam
from pyhaemo.spl import UnivariateKnotVector, TensorKnotVector, NDSpline

from solve import fit_constrained

import warnings
from typing import Literal


FORMAT = 'csv', 'swc'


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


def to_spline(centerline, radius, n_points=20, **fitkwargs):
    """
    Converts a single centerline and radius into spline representations.

    Args:
        centerline (numpy.ndarray): The centerline data points.
        radius (numpy.ndarray): The radius data points.
        n_points (int): Number of points for the spline representation.

    Returns:
        tuple: A tuple containing the centerline spline and radius spline.
    """

    fitkwargs.setdefault('lam0', 1e-2)

    kv_center = UnivariateKnotVector(np.linspace(0, 1, n_points)).to_tensor()

    # Compute the chord length parameter values of the centerline points
    xi_center = clparam(centerline)

    # Fit a univariate spline to the data
    centerline_spline = fit_constrained(kv_center, xi_center, centerline, **fitkwargs)
    radius_spline = fit_constrained(kv_center, xi_center, radius, **fitkwargs)

    return centerline_spline, radius_spline


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


def swc_converter(filename):
    """
        Converts an SWC file into a centerline and radius spline representation.
        The SWC file format contains information about the radius, centerline, and
        connectivity of a structure in a single file. This function processes the
        SWC file and extracts the necessary data to represent the structure as
        centerlines and radii splines.
        Args:
            filename (str): The path to the SWC file to be converted.
        Returns:
            tuple: A tuple containing:
                - centerlines_list (list): A list of centerline data extracted from the SWC file.
                - radii_list (list): A list of radius data corresponding to the centerlines.
    """
    process_input(filename, 'swc')

    # all the information has contained in one file in case of swc file
    points = []
    parent_map = {}  # To store parent-child relationships
    centerlines_list = []
    radii_list = []

    with open(filename, 'r') as f:
        for row in f:

            col = row.strip().split()
            if swc_input_check(row, col):
                continue

            points, parent_map = swc_data_extraction(col, points, parent_map)

    points = np.array(points)

    branches = []
    branch = []
    connetivity = []
    for i in range(2, len(parent_map)):
        if parent_map[i+1] - parent_map[i] == 1:
            branch.append(parent_map[i])
        elif parent_map[i+1] - parent_map[i] != 1:
            branch.append(parent_map[i])
            branch.append(i)
            branches.append(branch)
            branch = []

    valid_branches = []
    for i in range(len(branches)):
        if i % 2 == 0:
            valid_branches.append((np.array(branches[i])-1).tolist())
        else:
            connetivity.append((np.array(branches[i])-1).tolist())

    # Extract the centerline and radius data
    for i in range(len(valid_branches)):
        centerline = []
        radius = []
        for j in range(len(valid_branches[i])):
            centerline.append(points[valid_branches[i][j]][:3])
            radius.append(points[valid_branches[i][j]][3])
        if i != 0:
            centerline.insert(0, points[connetivity[i][0]][:3])
            radius.insert(0, points[connetivity[i][0]][3])

        centerlines_list.append(np.array(centerline))
        radii_list.append(np.array(radius))

    for i, (centerline, radius) in enumerate(zip(centerlines_list, radii_list)):
        centerline_spline, radius_spline = to_spline(centerline, radius)
        centerlines_list[i], radii_list[i] = centerline_spline, radius_spline

    return centerlines_list, radii_list, connetivity


def swc_input_check(row, columns):
    """
    Validates a row and its corresponding columns from an SWC file.
    This function performs two checks:
    1. If the row starts with a '#' character, it is considered a comment and
       the function returns True to indicate that the current iteration should be skipped.
    2. If the number of columns in the row is not equal to 7, it raises a ValueError
       indicating an invalid SWC format.
    Args:
        row (str): A single line from the SWC file.
        columns (list): A list of values parsed from the row.
    Returns:
        bool: True if the row should be skipped (e.g., it's a comment),
              False if the row passes validation and can be processed.
    Raises:
        ValueError: If the number of columns is not equal to 7.
    """

    # Check if row starts with '#', which will indicate to skip the current iteration
    if row.startswith('#'):
        return True  # Indicating to continue

    # Check if column length is invalid
    if len(columns) != 7:
        raise ValueError("Invalid SWC format.")

    return False  # No issues, proceed with the iteration


def swc_data_extraction(col, points, parent_map):
    """
    Extracts data from a single row of SWC file format and
    updates the points and parent_map dictionaries.
    Parameters:
        col (list): A list representing a row of SWC data. Expected to contain:
            col[0] (int): Node ID.
            col[2] (float): X-coordinate of the node.
            col[3] (float): Y-coordinate of the node.
            col[4] (float): Z-coordinate of the node.
            col[5] (float): Radius of the node.
            col[6] (int): Parent node ID.
        points (dict): A dictionary to store node data, where the
                      key is the node ID and the value is a tuple
                    containing the coordinates ([x, y, z]) and the radius.
        parent_map (dict): A dictionary to store the parent-child
                        relationships, where the key is the node ID
                        and the value is the parent node ID.
    Returns:
        tuple: A tuple containing:
            - points (dict): Updated dictionary with node data.
            - parent_map (dict): Updated dictionary with
                                parent-child relationships.
    """

    # Extract the relevant data
    node_id = int(col[0])
    x = float(col[2])
    y = float(col[3])
    z = float(col[4])
    radius = float(col[5])
    parent_id = int(col[6])

    points.append([x, y, z, radius])
    parent_map[node_id] = parent_id  # Store the parent ID for each node.

    return points, parent_map
