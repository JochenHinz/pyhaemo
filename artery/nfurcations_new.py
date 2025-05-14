#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:28:32 2023

@author: marcinno
"""

import array
import sys
from splico.spl import NDSplineArray
sys.path.append('/home/fabio/coronaries/core')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import copy
#from find_intersection import *
import create_video
from functools import lru_cache, wraps
sys.path.append('/home/fabio/splico/splico/geo')
sys.path.append('/home/fabio/splico/splico')
from splico.util import frozen
from splico.geo import CrossSectionMaker
from splico.mesh import rectilinear
import notwistframe_bif
from splico.util import np, _, clparam, normalize, frozen_cached_property
from splico.types import Immutable
from typing import Sequence, Self, Any, Optional, Callable


@lru_cache
def _cached_derivative(pol):
  return PolynomialCurve(pol.weights[1:] * np.arange(1, pol.order)[:, _])

def default_interval(fn):
  @wraps(fn)
  def wrapper(self, interval=None, **kwargs):
    if interval is None:
      interval = [0, 1]
    return fn(self, interval, **kwargs)
  return wrapper


class PolynomialCurve:

  def __init__(self, weights):
    self.weights = frozen(weights)
    assert len(self.weights.shape) == 2, NotImplementedError
    assert self.weights.shape[1:] == (3,), NotImplementedError
    self.order = len(self.weights)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self.weights.tobytes(), self.weights.shape))
    return self._hash

  def __eq__(self, other):
    if self.__class__ != other.__class__: return False
    if self.weights.shape != other.weights.shape: return False
    return self.__hash__() == other.__hash__()
  
  def _derivative(self):
    return _cached_derivative(self)

  @default_interval
  def find_approximate_arclength_breakpoints(self, interval, abscissae: np.ndarray, npoints: int):
      abscissae = np.asarray(abscissae)
      assert np.logical_and(abscissae >= 0, abscissae <= 1).all()
      assert npoints > 1
      xi = np.linspace(*interval, npoints)  # [xi0, xi1, ...]
      points = self(xi)  # [p0; p1; p2; ...]
      dx = np.array([0, *np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()])  # [0, L0, L0 + L1, ...]  Li = distance between pi+1 and pi
      dx = dx / dx[-1]  # total length of the discrete curve normalised
      argmins = np.argmin(np.abs(dx[:, _] - abscissae[_]), axis=0)  # find the positions in dx such that |dx[argmins[i]] - L * abscissae[i]| is minimised
      return np.asarray(argmins, dtype=int)  # return the xi that correspond to the entries in 'argmins'

  def derivative(self, n=1):
    if n == 0:
      return self
    return self._derivative().derivative(n=int(n-1))

# da chiedere bene a jochen cosa fa questo comando.
  def __call__(self, xi):
    if np.isscalar(xi):
      xi = np.array([xi])
    xi = xi[_]
    return (self.weights[:, _] * (xi ** (np.arange(self.order)[:, _]))[..., _]).sum(0)

def cubic_hermite_curve(p0, t0, p1, t1):
  weights = [p0, t0, -3 * p0 - 2 * t0 + 3 * p1 - t1, 2 * p0 + t0 - 2 * p1 + t1]
  return PolynomialCurve(weights)


def _compute_notwistframe(T, normal_first_slice, starting_frame):
    
  zero_thresh=1e-8
  assert zero_thresh >= 0
 
  normals = np.zeros(T.shape,dtype=np.float64)
  binormals = np.zeros(T.shape,dtype=np.float64)

  t0 = T[0]

  if starting_frame is not None and normal_first_slice is None:
      normals[0] = normalize(starting_frame[:,1])
      binormals[0] = normalize(starting_frame[:,0])
  if starting_frame is None and normal_first_slice is not None:
      normals[0] = normalize(normal_first_slice)
      binormals[0] = normalize((np.cross(t0, normals[0])))

  i = 1
  for told, tnew in zip(T, T[1:]):
    omega = np.cross(told, tnew)
    norm = (omega**2).sum()**.5
    if norm <= zero_thresh:
      binormal = binormals[i-1]
      normal = normals[i-1]
    else:
      omega = omega / norm
      phi = np.arccos(np.dot(told, tnew))
      binormal = (binormals[i-1] * np.cos(phi) + \
                 np.cross(omega, binormals[i-1]) * np.sin(phi) + \
                 omega * np.dot(omega, binormals[i-1]) * (1 - np.cos(phi)))
      normal = (normals[i-1] * np.cos(phi) + \
               np.cross(omega, normals[i-1]) * np.sin(phi) + \
               omega * np.dot(omega, normals[i-1]) * (1 - np.cos(phi)))

    binormals[i] = normalize(binormal)
    normals[i] = normalize(normal)
    i += 1
    
  return normals, binormals


class NoTwistFrame:

  
  def __init__(self, X, T):
    self.X = frozen(X)
    self.T = frozen(normalize(T))

    assert self.X.shape == self.T.shape
    assert self.X.shape[1:] == (3,)


  def compute_frame(self, normal_first_slice = None, starting_frame = None):

    normals, binormals = _compute_notwistframe(self.T , normal_first_slice, starting_frame)
    
    self._binormals = frozen(normalize(binormals))
    self._normals = frozen(normalize(normals))
  
    Rs = np.stack([ np.stack([n, bn, t], axis=1) for n, bn, t
                    in zip(self._binormals, self._normals, self.T) ], axis=0)
    
    self._Rs = frozen(Rs)
    
    
  def rotation_and_translation(self, Solutions, first = False):
    
    if first == True:
      j = 0
    else:
      j = -1
    
    Solutions = np.array(Solutions)

    for i in range(len(Solutions)):
        Solutions[i] = np.linalg.solve(np.linalg.inv(self._Rs[j]), Solutions[i])


    Solutions = Solutions + self.X[j]

    # Now the points are in the 3D space dimension.
    assert Solutions.shape[1:] == (3,)

    self.Solutions = Solutions


@lru_cache(maxsize=8)
def make_crosssection_maker(nelems: int):
  return CrossSectionMaker(nelems)


@lru_cache(maxsize=8)
def eval_disk(disk: NDSplineArray, lines_patch: int):
    
    points = [lines_patch, lines_patch] 
    mesh = []
    
    for spl in disk:
        mesh.append(spl.sample_mesh(rectilinear(points)))
        
    disk_final = np.concatenate((mesh[0].points, mesh[1].points, mesh[2].points, mesh[3].points, mesh[4].points))
    
    return disk_final


def patch_subdivision(disk: array, lines_patch: int):
    
    middle = lines_patch // 2
    patched_disk = disk.reshape(5, lines_patch, lines_patch,3)
    left_patch, bottom_patch, center_patch, top_patch, right_patch = patched_disk
    
    left = left_patch[middle, -1]
    bottom = bottom_patch[-1, middle]
    center = center_patch[middle, middle]
    top = top_patch[middle,-1]
    right = right_patch[-1, middle]
    
    return left, bottom, center, top, right


class Nfurcation(Immutable):
    
    def __init__(self,    
                 space: int, 
                 lines_patch: int,
                 radii: array,
                 tangents: array,
                 #points: array,
                 sol: Optional, 
                 maker: Optional = None):
        
        
        # distance between each section composing the bifurcation
        assert space > 0
        self.space = space
        
        # number of point on each line of the patch
        assert isinstance(lines_patch, int) and lines_patch > 0
        self.lines_patch = lines_patch
        
        # number of intersecting branches
        self.n_branches = sol.shape[0]
        
        # total points on a section
        self.total_points = sol.shape[1]
        assert self.total_points / (self.lines_patch**2 * 5) == 1,\
            'If this check fails, it means that the value lines_patch passed in the init is not the same used for the creation of the vessel sections.'
        
        assert sol.shape[2] == 3, "The vessel section has to be already positioned in the 3D space. Not accepted 2D arrays"
        self.sol = frozen(sol)
        
        assert radii.shape == (self.n_branches,)
        assert len(radii) == self.n_branches, 'For each disk must exist a radius information'
        assert np.all(radii > 0), 'The radius information must be positive' 
        self.radii = frozen(radii)
        
        assert tangents.shape == (self.n_branches, 3)
        assert len(tangents) == self.n_branches, 'For each disk must exist an own tangent'
        self.tangents = frozen(normalize(tangents)) 

        self.maker = make_crosssection_maker(self.lines_patch) if maker is None else maker
        
        self.left, self.bottom, self.center, self.top, self.right = map(frozen,zip(
            *(patch_subdivision(self.sol[i], self.lines_patch) for i in range(self.n_branches))))
        
    #    self.setup_sections()
        self.find_decomposition_points() 
        self.create_butterfly()
        self.create_butterfly_structure()
        self.butterfly_tangents()
        self.create_mesh() 
    
    #def setup_sections(self):
    
    def find_decomposition_points(self):
        
        curves_top = []
        curves_center = []
        curves_bottom = []
        
        self.dec_points = []
        self.dec_tangents = []

        #start_indices = np.arange(self.n_branches)
        #end_indices   = np.append(np.arange(1, self.n_branches), 0)

        #for start_index, end_index in zip(start_indices, end_indices):
    
        # TODO: make this scaling a parameter in the init and specific for each curve.
        alpha = 5
        for i in range(self.n_branches):
            
            start_index = i 
            end_index = i + 1 if i < (self.n_branches-1) else 0
            
            start_top = self.top[start_index]
            end_top      = self.top[end_index]
            
            start_center = self.center[start_index]
            end_center = self.center[end_index]
            
            start_bottom = self.bottom[start_index]
            end_bottom = self.bottom[end_index]
            
            start_tangent =  -alpha *self.radii[start_index] * self.tangents[start_index] 
            end_tangent   = alpha *self.radii[end_index] * self.tangents[end_index] 
            
            # Calculate the curve
            curves_top.append(cubic_hermite_curve(start_top, start_tangent, end_top, end_tangent))
            curves_bottom.append(cubic_hermite_curve(start_bottom, start_tangent, end_bottom, end_tangent))
            curves_center.append(cubic_hermite_curve(start_center, start_tangent, end_center, end_tangent))

            start = self.right[start_index]  
            end = self.left[end_index] 
            
            # Calculate the curve
            curve = (cubic_hermite_curve(start, start_tangent, end, end_tangent))
            
            self.dec_points.append((curve(0.5).reshape(3,)))
            self.dec_tangents.append(curve.derivative(n = 1)(0.5))  
               
        self.X_top = np.mean([curve(0.5).reshape(3,) for curve in curves_top], axis=0)
        self.X = np.mean([curve(0.5).reshape(3,) for curve in curves_center], axis=0)
        self.X_bottom = np.mean([curve(0.5).reshape(3,) for curve in curves_bottom], axis=0)

        
        self.vec_to_top = self.X_top - self.X 
        self.vec_to_bottom = self.X_bottom - self.X 
        self.vec_to_dec = self.dec_points -self.X
        
        
        #TODO: assert on the angles composed by the decomposition points, the angle between sp1 and sp2 must be counterclok!!        
        #for i in range(self.n_branches):
        #    angles = np.arccos(np.dot(normalize(self.vec_to_dec[i]),normalize(self.vec_to_dec[i+1])))
        #    direction = np.cross(normalize(self.vec_to_dec[i]),normalize(self.vec_to_dec[i+1]))
        
        
        # Sanity checks on n-furcations structure
        assert 1 - 10**-8 < np.abs(np.dot(normalize(self.X_top - self.X), normalize(self.X_bottom - self.X))) < 1 + 10**-8 ,\
        'X_top, X and X_top must lie on the same line.' 
        
        assert self.X_top.shape[0] == self.X.shape[0] == self.X_bottom.shape[0] == 3, 'All the decomposition points must be 3D'
                
        # Find the numeber of sections that the interpolation will need to mantain the same space between the sections of the single branch. 
        lenght = np.linalg.norm((self.X - self.center), axis = 1)
        
        n_slice = np.maximum(lenght // self.space, 3) # at least we want three sections (the begin, the end and one in the middle)
        
        self.t = [np.linspace(0, 1, int(slice)) for slice in n_slice]
    
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        
        # for i in range(self.n_branches):
        #     ax.scatter(self.left[i][0], self.left[i][1], self.left[i][2], s = 100, color = "red")
        #     ax.scatter(self.bottom[i][0], self.bottom[i][1], self.bottom[i][2], s=100, color="blue")
        #     ax.scatter(self.right[i][0], self.right[i][1], self.right[i][2], s=100, color="green")
        #     ax.scatter(self.top[i][0], self.top[i][1], self.top[i][2], s=100, color="purple")
        #     ax.scatter(self.center[i][0], self.center[i][1], self.center[i][2], s=100, color="orange")
        #     ax.scatter(self.dec_points[i][0], self.dec_points[i][1], self.dec_points[i][2], s = 200, color = "black")

            
        # ax.scatter(self.X_bottom[0],self.X_bottom[1],self.X_bottom[2],s = 150,color="purple")
        # ax.scatter(self.X_top[0],self.X_top[1],self.X_top[2],s = 150,color="purple")
        # ax.scatter(self.X[0],self.X[1],self.X[2],s = 150,color="purple")

        # ax.grid(False)
        # ax.view_init(172, 60)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])   
        # plt.show()


    def create_butterfly(self):
        
        vec_to_top = self.X_top - self.X 
        vec_to_bottom = self.X_bottom - self.X 
        vec_to_dec = self.dec_points -self.X
        
        dis_to_top = np.linalg.norm(vec_to_top)
        dis_to_bottom = np.linalg.norm(vec_to_bottom)
        dis_to_dec = np.linalg.norm(vec_to_dec, axis = 1)
        
        
        disc = []
        R = []
        for i in range(self.n_branches):
                sol_top = self.maker.make_disc(dis_to_dec[i], dis_to_top)
                disc.append(eval_disk(sol_top,  self.lines_patch))
    
                sol_bottom = self.maker.make_disc(dis_to_dec[i], dis_to_bottom)
                disc.append(eval_disk(sol_bottom, self.lines_patch))
                
                R.append(np.stack([normalize(vec_to_dec[i]), normalize(vec_to_top), normalize(np.cross(vec_to_dec[i], vec_to_top))], axis=1))  
                R.append(np.stack([-normalize(vec_to_dec[i]), normalize(vec_to_top), normalize(np.cross(-vec_to_dec[i], vec_to_top))], axis=1))        
    
        assert len(disc) == (2 * self.n_branches), f"for {self.n_branches}, we need {2*self.n_branches}."
        assert len(R) == (2 *self.n_branches)

        
        self.disc = []
        self.R = []
        for i in range(self.n_branches):
            self.disc.append([disc[(i * 2) - 2], disc[(i * 2) - 1], disc[i * 2], disc[(i * 2) + 1]])
            self.R.append([R[(i * 2) - 1], R[i * 2]])
          

    def create_butterfly_structure(self):

        def create_interface(total_points, disks, tol):
    
            # disk 1 is for the quadrant left top, disk2 is for the quadrant left bottom
            # disk 3 is for the quadrant right top, disk4 is for the quadrant right bottom 
            disk_1,disk_2,disk_3, disk_4 =  disks
    
            inter = []
        
            for i in range(total_points):
                if disk_3[i,0] > tol and disk_3[i,1] > tol:
                    inter.append(disk_3[i])
                elif disk_1[i,0] < tol and disk_1[i,1] > tol:
                    inter.append(disk_1[i])
                elif disk_2[i,0] < tol and disk_2[i,1] < tol:
                    inter.append(disk_2[i])
                elif disk_4[i,0] > tol and disk_4[i,1] < tol:
                    inter.append(disk_4[i])
                else:
                    raise Exception('bifurcation crashing at the interface')
        
            inter = np.array(inter)
            assert inter.shape[0] == total_points, "The number of points of the interface disk (inter) \
                                                        is not equal to the other disks. Wrong dimension."
            assert inter.shape[1] == 3, "The dimensionality of the coordinate must be 3"
        
            return inter        
        
        def positioning_butterfly(total_points, inter, rotation_matrices, center_bif, tol):
        
            R1, R2 = rotation_matrices
            
            for i in range(total_points):
            
                if inter[i,0] > tol :
                    inter[i] = np.linalg.solve(np.linalg.inv(R2), inter[i])
                else:
                    inter[i] = np.linalg.solve(np.linalg.inv(R1), inter[i])
            
            inter = inter + center_bif

            assert len(inter) == total_points, \
            "The rotation and traslation has affected somehow the dimension of the interface butterfly structure."

            return inter
              
        self.inter = []
        self.inter2D_list = []
        for i in range(self.n_branches):
            self.inter2D = create_interface(self.total_points, self.disc[i], tol = 10**-12)
            self.inter2D_list.append(self.inter2D)
            self.inter.append(positioning_butterfly(self.total_points, self.inter2D, self.R[i], self.X, tol = 10**-12))

        #create_video.create_video_butterfly(self.sol[0], self.sol[1], self.sol[2], self.dec_points[0], self.dec_points[1], self.dec_points[2],\
        #    self.X_bottom, self.X_top, self.X, self.inter[0], self.inter[1], self.inter[2], filename='animation.gif', show=True)
    
    # to be finished
    def butterfly_tangents(self):
        
        import ipdb
        ipdb.set_trace()        
        max_right = []
        max_left = []
        self.middle = int(self.lines_patch/2 - 1/2)
        for i in range (self.n_branches):
            max_right.append(self.inter[i].reshape(5,self.lines_patch,self.lines_patch,3)[4,-1, self.middle])
            max_left.append(self.inter[i].reshape(5,self.lines_patch,self.lines_patch,3)[0, self.middle, -1])
        
        
        self.t_inter = []
        for i in range (self.n_branches):
            for j in range(self.total_points):

                if self.inter2D_list[i][j,0] > 10**(-12):
                     self.t_inter.append(self.inter2D_list[i][j,0] * (self.dec_tangents[i])/2 + (max_right[i][0] - self.inter2D_list[i][j,0]) * (+self.dec_tangents[i]-self.dec_tangents[-1+i])/2)
                else:
                     self.t_inter.append(np.abs(self.inter2D_list[i][j,0]) * (-self.dec_tangents[-1+i])/2 + (np.abs(max_left[i][0]) - np.abs(self.inter2D_list[i][j,0])) * (+self.dec_tangents[i] - self.dec_tangents[-1+i])/ 2)


        self.t_inter = np.array(self.t_inter).reshape(self.n_branches,self.total_points,3)


    def create_mesh(self, hermite = True):    
        
        alpha = 2
        
        for i in range(self.n_branches): 
            
            start_tangent =  -alpha *self.radii[i] * self.tangents[i] 
            end_tangent   = 0.5 * normalize(self.t_inter[i])
            
            self._3D_interp_sol = []
            
            if hermite:
                interp = hermite_mesh(self.sol[i], self.inter[i], start_tangent, end_tangent, len(self.t[i]))
                self._3D_interp_sol.append(interp)
            else:
                for t in (self.t[i]):
                   interp =  (1 - t) * self.sol[i] +  t * self.inter[i]

                   self._3D_interp_sol.append(interp)
                    

            mesh = np.array(self._3D_interp_sol).reshape(len(self.t[i]), 5,self.lines_patch,self.lines_patch,3)  
        
            from mesh_tester import create_gmsh
            create_gmsh([*mesh], filename=f'bif_{i}.vtk')
            # TODO: check on the mesh validity, else make it with diffrent hermite scaling tangents.
        
def hermite_mesh(sol_ini, sol_fin, tan_ini, tan_fin, t_hermite):
        
       interp_sol = []
      
       tangent_ini = tan_ini*np.ones(sol_ini.shape)
       tangent_fin = tan_fin*np.ones(sol_ini.shape)

       for j in range(t_hermite):
            new_disk = np.zeros((sol_ini.shape))
           
            for i in range(sol_ini.shape[0]):
               
               curve = cubic_hermite_curve(sol_ini[i], tangent_ini[i], sol_fin[i], tangent_fin[i])
               t = np.linspace(0,1,t_hermite) 
               npoints = 1000
               argmins = curve.find_approximate_arclength_breakpoints(abscissae=t, npoints=npoints)
               t = argmins/argmins[-1]

               H_1 = 2*t[j]**3 - 3*t[j]**2 +1
               H_2 = t[j]**3 - 2*t[j]**2 + t[j]
               H_3 = -2*t[j]**3 + 3*t[j]**2
               H_4 = t[j]**3 - t[j]**2
               
               new_disk[i] = H_1 * sol_ini[i] + H_2 *tangent_ini[i] + H_3 * sol_fin[i] +  H_4 * tangent_fin[i]

            interp_sol.append(copy.copy(new_disk))
                  
       interp_sol = np.array(interp_sol) 
         
       return interp_sol   
        
        
def test_case():
    
    # number of evaluation points on avery line of the mesh circle
    numpoints = 11
    
    maker = make_crosssection_maker(numpoints)
    
    # angle between the principal direction, see the plot above.
    alpha_main_center = np.pi/2
    alpha_slave_1_center = np.pi/20
    alpha_slave_2_center = np.pi/20

    alpha_slave_3_center = np.pi/3
    alpha_slave_4_center = np.pi/3   
    # create the solution of the three branch just before starting with the bifurcation
    radius = 1.5
    sol = maker.make_disc(radius, radius)
    sol_main = eval_disk(sol, numpoints)
    sol_slave_1 = eval_disk(sol, numpoints)
    sol_slave_2 = eval_disk(sol, numpoints)
    sol_slave_3 = eval_disk(sol, numpoints)
    sol_slave_4 = eval_disk(sol, numpoints)   
    
    radii = np.stack((radius,radius,radius,radius,radius),axis = 0)
    # centerlines, need to use np.flip, otherwise the disc are inserted on the center of the bifurcation and not just before.
    
    n = 8
    points_main = np.stack(([0]*n, [0]*n, np.linspace(-5,0,n)), axis = 1)
    
    points_slave_1 = np.flip(np.stack((np.linspace(0, -5*np.cos(alpha_slave_1_center) ,n), [0]*n, np.linspace(0, -5*np.sin(alpha_slave_1_center), n)), axis = 1),axis = 0)
    points_slave_2 = np.flip(np.stack((np.linspace(0, 5*np.cos(alpha_slave_2_center) ,n),  [0]*n, np.linspace(0, -5*np.sin(alpha_slave_2_center), n)), axis = 1), axis = 0)
    points_slave_3 = np.flip(np.stack((np.linspace(0, -5*np.cos(alpha_slave_3_center) ,n), [0]*n, np.linspace(0, 6*np.sin(alpha_slave_3_center), n)), axis = 1),axis = 0)
    points_slave_4 = np.flip(np.stack((np.linspace(0, 5*np.cos(alpha_slave_4_center) ,n), [0]*n, np.linspace(0, 6*np.sin(alpha_slave_4_center), n)), axis = 1), axis = 0)
          
    #XXX All the tangents must points outward!
    tangent_main = normalize(np.stack(([0]*n, [0]*n, [-1]*n), axis = 1))
    tangent_slave_1 = -normalize(np.stack(([1]*n, [0]*n, [0.5]*n), axis = 1))
    tangent_slave_2 = -normalize(np.stack(([-1]*n, [-0.7]*n, [0.5]*n), axis = 1))
    tangent_slave_3 = -normalize(np.stack(([1]*n, [-0.7]*n, [-1]*n), axis = 1))
    tangent_slave_4 = -normalize(np.stack(([-1]*n, [0.7]*n, [-1]*n), axis = 1)) 
    
    vec1 = normalize(tangent_main[0])
    vec2 = normalize(tangent_slave_1[0])
    vec3 = normalize(np.cross(vec1,vec2))
    
    normal = normalize(vec3 - (np.dot(vec3,normalize(vec1)))*normalize(vec1)/(np.dot(vec1, vec1)))
        
    # of the bifurcation geometry
    frame_main = notwistframe_bif.NoTwistFrame(points_main, tangent_main)
    frame_main.compute_frame(normal_first_slice= normal, starting_frame = None)
    frame_main.rotation_and_translation(sol_main, first = True)
    
    frame_slave_1 = notwistframe_bif.NoTwistFrame(points_slave_1, tangent_slave_1)
    frame_slave_1.compute_frame(normal_first_slice=normal, starting_frame = None)
    frame_slave_1.rotation_and_translation(sol_slave_1, first = True)
    
    frame_slave_2 = notwistframe_bif.NoTwistFrame(points_slave_2, tangent_slave_2)
    frame_slave_2.compute_frame(normal_first_slice=normal, starting_frame = None)
    frame_slave_2.rotation_and_translation(sol_slave_2, first = True)
   
    frame_slave_3 = notwistframe_bif.NoTwistFrame(points_slave_3, tangent_slave_3)
    frame_slave_3.compute_frame(normal_first_slice=normal, starting_frame = None)
    frame_slave_3.rotation_and_translation(sol_slave_3, first = True)
    
    frame_slave_4 = notwistframe_bif.NoTwistFrame(points_slave_4, tangent_slave_4)
    frame_slave_4.compute_frame(normal_first_slice=normal, starting_frame = None)
    frame_slave_4.rotation_and_translation(sol_slave_4, first = True)


    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(frame_main.Solutions[:,0], frame_main.Solutions[:,1], frame_main.Solutions[:,2],s = 100, color = "red")
    ax.scatter(frame_main.Solutions[10,0], frame_main.Solutions[10,1], frame_main.Solutions[10,2],s = 150, color = "black")
    ax.scatter(frame_slave_1.Solutions[:,0], frame_slave_1.Solutions[:,1], frame_slave_1.Solutions[:,2],s = 100, color = "yellow")
    ax.scatter(frame_slave_1.Solutions[10,0], frame_slave_1.Solutions[10,1], frame_slave_1.Solutions[10,2],s = 150, color = "black")
    ax.scatter(frame_slave_2.Solutions[:,0], frame_slave_2.Solutions[:,1], frame_slave_2.Solutions[:,2],s = 100, color = "green")
    ax.scatter(frame_slave_2.Solutions[10,0], frame_slave_2.Solutions[10,1], frame_slave_2.Solutions[10,2],s = 150, color = "black")
    ax.quiver(frame_main.Solutions[:,0], frame_main.Solutions[:,1], frame_main.Solutions[:,2], tangent_main[0][0], tangent_main[0][1],tangent_main[0][2])
    ax.quiver(frame_slave_1.Solutions[:,0], frame_slave_1.Solutions[:,1], frame_slave_1.Solutions[:,2], tangent_slave_1[0][0], tangent_slave_1[0][1],tangent_slave_1[0][2])
    ax.quiver(frame_slave_2.Solutions[:,0], frame_slave_2.Solutions[:,1], frame_slave_2.Solutions[:,2], tangent_slave_2[0][0], tangent_slave_2[0][1],tangent_slave_2[0][2])

    #plt.show()
    
    #XXX everything must be given in counterclockwise. #XXX All the tangents must points outward!
    tangents = np.stack((tangent_main[0],tangent_slave_2[0], tangent_slave_4[0], tangent_slave_3[0], tangent_slave_1[0]), axis = 0)
    sol = np.stack((frame_main.Solutions,  frame_slave_2.Solutions, frame_slave_4.Solutions,frame_slave_3.Solutions, frame_slave_1.Solutions), axis = 0)
    
    bif = Nfurcation(space = 0.8, lines_patch = numpoints , radii = radii, tangents = tangents, sol = sol)    

if __name__ == '__main__':

    test_case()  
    


    
