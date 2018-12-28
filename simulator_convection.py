# ==========================================
# Code created by Leandro Marques at 12/2018
# Gesar Search Group
# State University of the Rio de Janeiro
# e-mail: marquesleandro67@gmail.com
# ==========================================

# FEM Simulator - Convection



# =======================
# Importing the libraries
# =======================

import sys
sys.path.insert(0, '../lib_class')

from tqdm import tqdm
from time import time

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

import import_msh
import assembly
import bc_apply
import semi_lagrangian
import export_vtk




print '------------'
print 'IMPORT MESH:'
print '------------'

start_time = time()

name_mesh = 'malha_convection.msh'
number_equation = 1
mesh = import_msh.Linear('../mesh',name_mesh,number_equation)
mesh.ien()
mesh.coord()


end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""



# ==========
# Parameters
# ==========

# Time
dt = 0.05
nt = 250
theta = 1.0

# Nondimensional Numbers
Re = 1000.0
Sc = 1.0

# ---------------- PARAMETERS ------------------
print '-----------------------------'
print 'PARAMETERS OF THE SIMULATION:'
print '-----------------------------'
print 'Mesh: %s' %name_mesh
print 'Number of nodes: %s' %mesh.npoints
print 'Number of elements: %s' %mesh.nelem
print 'Time step: %s' %dt
print 'Number of time iteration: %s' %nt
print 'Time scheme (theta): %s' %theta
print 'Reynolds number: %s' %Re
print 'Schmidt number: %s' %Sc
print ""
# -----------------------------------------------


print '--------'
print 'ASSEMBLY:'
print '--------'

start_time = time()

Kxx, Kxy, Kyx, Kyy, K, M, MLump, Gx, Gy = assembly.Linear(mesh.npoints, mesh.nelem, mesh.IEN, mesh.x, mesh.y)

#LHS_c0 = (sps.lil_matrix.copy(M)/dt) + (theta/Re)*sps.lil_matrix.copy(K)
LHS_c0 = (sps.lil_matrix.copy(M)/dt)


end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""


print '--------------------------------'
print 'INITIAL AND BOUNDARY CONDITIONS:'
print '--------------------------------'

start_time = time()

# --------- Boundaries conditions --------------------
condition_concentration = bc_apply.Convection(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
condition_concentration.neumann_condition(mesh.neumann_edges[1])
condition_concentration.dirichlet_condition(mesh.dirichlet_pts[1])
condition_concentration.gaussian_elimination(LHS_c0,mesh.neighbors_nodes)
# ----------------------------------------------------

# --------- Initial condition ------------------------
condition_concentration.initial_condition()
c = np.copy(condition_concentration.c)
vx = np.copy(condition_concentration.vx)
vy = np.copy(condition_concentration.vy)
# ----------------------------------------------------



end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""



print '----------------------------'
print 'SOLVE THE LINEARS EQUATIONS:'
print '----------------------------'
print ""

for t in tqdm(range(0, nt)):

 # ------------------------ Export VTK File ---------------------------------------
 save = export_vtk.Linear(mesh.x,mesh.y,mesh.IEN,mesh.npoints,mesh.nelem,c,c,c,vx,vy)
 save.saveVTK('/home/marquesleandro/results/convection_lagrangian2','coronary%s' %t)
 # --------------------------------------------------------------------------------

 # ------------------------ Solver - Taylor Galerkin ------------------------------
# A = np.copy(M)/dt
# concentration_RHS = sps.lil_matrix.dot(A,c) - np.multiply(vx,sps.lil_matrix.dot(Gx,c))\
#                                             - np.multiply(vy,sps.lil_matrix.dot(Gy,c))\
#                   - (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(Kxx,c))\
#                                            + np.multiply(vy,sps.lil_matrix.dot(Kyx,c))))\
#                   - (dt/2.0)*np.multiply(vy,(np.multiply(vx,sps.lil_matrix.dot(Kxy,c))\
#                                            + np.multiply(vy,sps.lil_matrix.dot(Kyy,c))))
# 
# concentration_RHS = np.multiply(concentration_RHS,condition_concentration.bc_2)
# concentration_RHS = concentration_RHS + condition_concentration.bc_dirichlet
# 
# c = scipy.sparse.linalg.cg(condition_concentration.LHS,concentration_RHS,c, maxiter=1.0e+05, tol=1.0e-05)
# c = c[0].reshape((len(c[0]),1))
 # --------------------------------------------------------------------------------

 # ------------------------ Solver - Semi Lagrangian ------------------------------
 x_d = mesh.x - vx*dt
 y_d = mesh.y - vy*dt

 c_d = semi_lagrangian.Linear_2D(mesh.npoints, mesh.IEN, mesh.x, mesh.y, x_d, y_d, mesh.neighbors_elements, c)

 A = np.copy(M)/dt
 concentration_RHS = sps.lil_matrix.dot(A,c_d)
 
 concentration_RHS = np.multiply(concentration_RHS,condition_concentration.bc_2)
 concentration_RHS = concentration_RHS + condition_concentration.bc_dirichlet

 c = scipy.sparse.linalg.cg(condition_concentration.LHS,concentration_RHS,c, maxiter=1.0e+05, tol=1.0e-05)
 c = c[0].reshape((len(c[0]),1))
 # --------------------------------------------------------------------------------