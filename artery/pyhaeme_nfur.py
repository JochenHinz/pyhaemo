#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jul 31 10:28:32 2023

@author: marcinno
"""

from pyhaemo.spl import NDSpline
from pyhaemo.util import np, normalize
from pyhaemo.geo.interp import cubic_hermite_interpolation as cubic_hermite_curve


def find_butterfly_line(radii, points, frames):

 tangents = frames[:, -1]

 left, right, center, top, bottom = [], [], [], [], []
 n_branches = len(points)

 for i in range(n_branches):
   left.append(points[i] - radii[i] * frames[i][:, 1])
   right.append(points[i] + radii[i] * frames[i][:, 1])
   center.append(points[i])
   top.append(points[i] + radii[i] * frames[i][:, 0])
   bottom.append(points[i] - radii[i] * frames[i][:, 0])

 curves_top = []
 curves_center = []
 curves_bottom = []
 dec_points = []
 dec_tangents = []

 # TODO: make this scaling a parameter specific for each curve.
 alpha = 2
 for i in range(n_branches):

     start_index = i
     end_index = i + 1 if i < (n_branches-1) else 0

     start_top = NDSpline([], [top[start_index]])
     end_top = NDSpline([], [top[end_index]])

     start_center = NDSpline([], [center[start_index]])
     end_center = NDSpline([], [center[end_index]])

     start_bottom = NDSpline([], [bottom[start_index]])
     end_bottom = NDSpline([], [bottom[end_index]])

     start_tangent = -alpha*radii[start_index] * tangents[start_index]
     end_tangent = alpha * radii[end_index] * tangents[end_index]

     # Calculate the curve
     curves_top.append(cubic_hermite_curve(start_top, end_top,
                                           start_tangent, end_tangent))
     curves_bottom.append(cubic_hermite_curve(start_bottom, end_bottom, start_tangent, end_tangent))
     curves_center.append(cubic_hermite_curve(start_center, end_center, start_tangent, end_tangent))

     start = NDSpline([], [[right[start_index]]])
     end = NDSpline([], [[left[end_index]]])

     # Calculate the curve
     curve = cubic_hermite_curve(start, end,
                                 start_tangent, end_tangent)

     dec_points.append((curve([0.5]).reshape(3,)))
     dec_tangents.append(curve([0.5], dx=1))

 X_top = np.mean([curve([0.5]).reshape(3,) for curve in curves_top], axis=0)
 X = np.mean([curve([0.5]).reshape(3,) for curve in curves_center], axis=0)
 X_bottom = np.mean([curve([0.5]).reshape(3,) for curve in curves_bottom], axis=0)

 assert 1 - 10**-8 < np.abs(np.dot(normalize(X_top - X), normalize(X_bottom - X))) < 1 + 10**-8 , 'X_top, X and X_top must lie on the same line.'
 assert X_top.shape[0] == X.shape[0] == X_bottom.shape[0] == 3, 'All the decomposition points must be 3D'

 return X_top, X, X_bottom

def plot_points(X_top, X, X_bottom, dec_points, left, bottom, right, top):

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(left)):
            ax.scatter(*left[i], s=100, color="red")
            ax.scatter(*bottom[i], s=100, color="blue")
            ax.scatter(*right[i], s=100, color="green")
            ax.scatter(*top[i], s=100, color="purple")
            ax.scatter(*dec_points[i], s=200, color="black")
        ax.scatter(*X_bottom, s=150, color="purple")
        ax.scatter(*X_top, s=150, color="purple")
        ax.scatter(*X, s=150, color="purple")

        ax.grid(False)
        ax.view_init(172, 60)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.show()

if __name__ == '__main__':

    radii = [.2, .2, .2, .2]

    points = np.array([[0, 0, 0],
                       [1, 1, 0],
                       [2, 0, 0],
                       [-1, 1, 0]])

    frames = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
              np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
              np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
              np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])]

    left, bottom, right, top, X_top, X, X_bottom, dec_points = find_butterfly_line(radii, points, frames)
    plot_points(X_top, X, X_bottom, dec_points, left, bottom, right, top)
