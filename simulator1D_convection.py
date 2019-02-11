# ==========================================
# Code created by Leandro Marques at 02/2019
# Gesar Search Group
# State University of the Rio de Janeiro
# e-mail: marquesleandro67@gmail.com
# ==========================================

# FEM Simulator - Convection 1D



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

name_mesh = 'malha_1D.msh'
number_equation = 1
mesh = import_msh.Linear1D('../mesh',name_mesh,number_equation)
mesh.coord()
mesh.ien()


end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""

# ==========
# Parameters
# ==========

# Time
CFL = 0.5
dt = float(CFL*mesh.length_min)
nt = 2500


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
print 'Reynolds number: %s' %Re
print 'Schmidt number: %s' %Sc
print ""
# -----------------------------------------------


print '--------'
print 'ASSEMBLY:'
print '--------'

start_time = time()

K, M, G = assembly.Linear1D(mesh.GL, mesh.npoints, mesh.nelem, mesh.IEN, mesh.x)

LHS_c0 = (sps.lil_matrix.copy(M)/dt)


end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""


print '--------------------------------'
print 'INITIAL AND BOUNDARY CONDITIONS:'
print '--------------------------------'

start_time = time()

# --------- Boundaries conditions --------------------
condition_concentration = bc_apply.Convection1D(mesh.nphysical,mesh.npoints,mesh.x)
condition_concentration.neumann_condition(mesh.neumann_pts[1])
condition_concentration.dirichlet_condition(mesh.dirichlet_pts[1])
condition_concentration.gaussian_elimination(LHS_c0,mesh.neighbors_nodes)
# ----------------------------------------------------

# --------- Initial condition ------------------------
condition_concentration.initial_condition()
c = np.copy(condition_concentration.c)
vx = np.copy(condition_concentration.vx)
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
 save = export_vtk.Linear1D(mesh.x,mesh.IEN,mesh.npoints,mesh.nelem,c,c,c,vx,vx)
 save.saveVTK('/home/marquesleandro/results/convection_semilagrangian','convection%s' %t)
 # --------------------------------------------------------------------------------

 # ------------------------ Solver - Semi Lagrangian ------------------------------
 x_d = mesh.x - vx*dt

 #c_d = semi_lagrangian.Linear1D_v2(mesh.npoints, mesh.nelem, mesh.IEN, mesh.x, x_d, c)
 c_d = semi_lagrangian.Linear1D(mesh.npoints, mesh.IEN, mesh.x, x_d, mesh.neighbors_elements, c)

 A = np.copy(M)/dt
 concentration_RHS = sps.lil_matrix.dot(A,c_d)
 
 concentration_RHS = np.multiply(concentration_RHS,condition_concentration.bc_2)
 concentration_RHS = concentration_RHS + condition_concentration.bc_dirichlet

 c = scipy.sparse.linalg.cg(condition_concentration.LHS,concentration_RHS,c, maxiter=1.0e+05, tol=1.0e-05)
 c = c[0].reshape((len(c[0]),1))
 c[0] = c[1]
# --------------------------------------------------------------------------------

