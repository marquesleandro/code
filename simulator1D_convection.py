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
import solver
import export_vtk
import relatory



print '''
               COPYRIGHT                    
 ======================================
 Simulator: %s
 created by Leandro Marques at 02/2019
 e-mail: marquesleandro67@gmail.com
 Gesar Search Group
 State University of the Rio de Janeiro
 ======================================
''' %sys.argv[0]




print ' ------'
print ' INPUT:'
print ' ------'

print ""
mesh_name = (raw_input(" Enter mesh name (.msh): ") + '.msh')
equation_number = int(raw_input(" Enter equation number: "))
print ""

Re = float(raw_input(" Enter Reynolds Number (Re): "))
Sc = float(raw_input(" Enter Schmidt Number (Sc): "))
print ""

nt = int(raw_input(" Enter number of time interations (nt): "))
directory_name = raw_input(" Enter folder name to save simulations: ")

print ""
print ' (1) - Taylor Galerkin'
print ' (2) - Semi Lagrangian'
scheme_option = int(raw_input(" Enter simulation scheme option above: "))
print ""

# INCLUIR O INPUT DO SIMULATOR2D
polynomial_order = 'teste'
gausspoints = 1


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
import_mesh_time = end_time - start_time
print ' time duration: %.1f seconds' %import_mesh_time
print ""




print ' ---------'
print ' ASSEMBLY:'
print ' ---------'

start_time = time()

K, M, G = assembly.Linear1D(mesh.GL, mesh.npoints, mesh.nelem, mesh.IEN, mesh.x)

LHS_c0 = (sps.lil_matrix.copy(M)/dt)

end_time = time()
assembly_time = end_time - start_time
print ' time duration: %.1f seconds' %assembly_time
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
bc_apply_time = end_time - start_time
print ' time duration: %.1f seconds' %bc_apply_time
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

start_time = time()
for t in tqdm(range(0, nt)):
 
 # ------------------------ Export VTK File --------------------------------------
 save = export_vtk.Linear1D(mesh.x,mesh.IEN,mesh.npoints,mesh.nelem,c,c,c,vx,vx)
 save.create_dir(directory_name)
 save.saveVTK(directory_name + str(t))
 # -------------------------------------------------------------------------------

 # -------------------------------- Solver ---------------------------------------
 # Taylor Galerkin
 if scheme_option == 1:
  scheme = solver.SemiImplicit_convection_diffusion1D(scheme_option)
  scheme.taylor_galerkin(c, vx, dt, M, K, G, condition_concentration.LHS, condition_concentration.bc_dirichlet, condition_concentration.bc_2)
  c = scheme.c

 # Semi Lagrangian
 elif scheme_option == 2:
  scheme = solver.SemiImplicit_convection_diffusion1D(scheme_option)
  scheme.semi_lagrangian(mesh.npoints, mesh.neighbors_elements, mesh.IEN, mesh.x, mesh.y, vx, vy, dt, c, M, condition_concentration.LHS, condition_concentration.bc_dirichlet, condition_concentration.bc_2)
  c = scheme.c
 # -------------------------------------------------------------------------------



 # ----------------------- Solver - Taylor Galerkin ------------------------------
# scheme = 'Taylor Galerkin'
# A = np.copy(M)/dt
# concentration_RHS = sps.lil_matrix.dot(A,c) - np.multiply(vx,sps.lil_matrix.dot(G,c))\
#                   - (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(K,c))))
# 
# concentration_RHS = np.multiply(concentration_RHS,condition_concentration.bc_2)
# concentration_RHS = concentration_RHS + condition_concentration.bc_dirichlet
# 
# c = scipy.sparse.linalg.cg(condition_concentration.LHS,concentration_RHS,c, maxiter=1.0e+05, tol=1.0e-05)
# c = c[0].reshape((len(c[0]),1))
 # -------------------------------------------------------------------------------


 # ------------------------ Solver - Semi Lagrangian -----------------------------
# scheme = 'Semi Lagrangian'
# c_d = semi_lagrangian.Linear1D(mesh.npoints, mesh.neighbors_elements, mesh.IEN, mesh.x, vx, dt, c)
#
# A = np.copy(M)/dt
# concentration_RHS = sps.lil_matrix.dot(A,c_d)
# 
# concentration_RHS = np.multiply(concentration_RHS,condition_concentration.bc_2)
# concentration_RHS = concentration_RHS + condition_concentration.bc_dirichlet
#
# c = scipy.sparse.linalg.cg(condition_concentration.LHS,concentration_RHS,c, maxiter=1.0e+05, tol=1.0e-05)
# c = c[0].reshape((len(c[0]),1))
 # -------------------------------------------------------------------------------

end_time = time()
solution_time = end_time - start_time

relatory.export(directory_name, sys.argv[0], scheme.scheme_name, mesh_name, equation_number, mesh.npoints, mesh.nelem, mesh.length_min, dt, nt, Re, Sc, import_mesh_time, assembly_time, bc_apply_time, solution_time, polynomial_order, gausspoints)
