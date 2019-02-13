# =======================
# Importing the libraries
# =======================

import sys
directory = '/home/marquesleandro/lib_class'
sys.path.insert(0, directory)

from tqdm import tqdm
from time import time

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

import search_file
import import_msh
import assembly
import bc_apply
import semi_lagrangian
import export_vtk




print ""
print " Simulator: %s" %sys.argv[0]

print '''
 ===============================================
 Simulator created by Leandro Marques at 02/2019
 e-mail: marquesleandro67@gmail.com
 Gesar Search Group
 State University of the Rio de Janeiro
 ===============================================
'''
print ""




print ' ------'
print ' INPUT:'
print ' ------'

print ""
mesh_name = (raw_input(" Enter mesh name (.msh): ") + '.msh')
equation_number = int(raw_input(" Enter equation number: "))
nt = int(raw_input(" Enter number of time interations (nt): "))
Re = float(raw_input(" Enter Reynolds Number (Re): "))
Sc = float(raw_input(" Enter Schmidt Number (Sc): "))
directory_name = raw_input(" Enter folder name to save simulations: ")
print ""




print ' ------------'
print ' IMPORT MESH:'
print ' ------------'

start_time = time()

directory = search_file.Find(mesh_name)
if directory == 'File not found':
 sys.exit()

mesh = import_msh.Linear1D(directory,mesh_name,equation_number)
mesh.coord()
mesh.ien()

CFL = 0.5
dt = float(CFL*mesh.length_min)

end_time = time()
print ' time duration: %.1f seconds' %(end_time - start_time)
print ""




print ' ---------'
print ' ASSEMBLY:'
print ' ---------'

start_time = time()

K, M, G = assembly.Linear1D(mesh.GL, mesh.npoints, mesh.nelem, mesh.IEN, mesh.x)

LHS_c0 = (sps.lil_matrix.copy(M)/dt)

end_time = time()
print ' time duration: %.1f seconds' %(end_time - start_time)
print ""




print ' --------------------------------'
print ' INITIAL AND BOUNDARY CONDITIONS:'
print ' --------------------------------'

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
print ' time duration: %.1f seconds' %(end_time - start_time)
print ""




print ' -----------------------------'
print ' PARAMETERS OF THE SIMULATION:'
print ' -----------------------------'

print ' Mesh: %s' %mesh_name
print ' Number of equation: %s' %equation_number
print ' Number of nodes: %s' %mesh.npoints
print ' Number of elements: %s' %mesh.nelem
print ' Smallest edge length: %f' %mesh.length_min
print ' Time step: %s' %dt
print ' Number of time iteration: %s' %nt
print ' Reynolds number: %s' %Re
print ' Schmidt number: %s' %Sc
print ""




print ' ----------------------------'
print ' SOLVE THE LINEARS EQUATIONS:'
print ' ----------------------------'
print ""
print ' Saving simulation in %s' %directory_name
print ""

for t in tqdm(range(0, nt)):
 
 # ------------------------ Export VTK File ---------------------------------------
 save = export_vtk.Linear1D(mesh.x,mesh.IEN,mesh.npoints,mesh.nelem,c,c,c,vx,vx)
 save.create_dir(directory_name)
 save.saveVTK(directory_name + str(t))
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

