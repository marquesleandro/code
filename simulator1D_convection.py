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

print ' (1) - Linear Element'
print ' (2) - Quadratic Element'
print ' (3) - Cubic Element'
polynomial_option = int(raw_input(" Enter polynomial degree option above: "))
print ""

print '  3 Gauss Points'
print '  4 Gauss Points'
print '  5 Gauss Points'
print ' 10 Gauss Points'
gausspoints = int(raw_input(" Enter Gauss Points Number option above: "))
print ""

nt = int(raw_input(" Enter number of time interations (nt): "))
directory_name = raw_input(" Enter folder name to save simulations: ")

print ""
print ' (1) - Taylor Galerkin'
print ' (2) - Semi Lagrangian Linear'
print ' (3) - Semi Lagrangian Quadratic'
scheme_option = int(raw_input(" Enter simulation scheme option above: "))
print ""


print ' ------------'
print ' IMPORT MESH:'
print ' ------------'

start_time = time()

directory = search_file.Find(mesh_name)
if directory == 'File not found':
 sys.exit()

if polynomial_option == 1:
 mesh = import_msh.Linear1D(directory,mesh_name,equation_number)
 mesh.coord()
 mesh.ien()

elif polynomial_option == 2:
 mesh = import_msh.Quad1D(directory,mesh_name,equation_number)
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

K, M, G, polynomial_order = assembly.Element1D(polynomial_option, mesh.GL, mesh.npoints, mesh.nelem, mesh.IEN, mesh.x, gausspoints)

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



# Taylor Galerkin
if scheme_option == 1:
 
 start_time = time()
 for t in tqdm(range(0, nt)):
 
  # ------------------------ Export VTK File --------------------------------------
  save = export_vtk.Linear1D(mesh.x,mesh.IEN,mesh.npoints,mesh.nelem,c,c,c,vx,vx)
  save.create_dir(directory_name)
  save.saveVTK(directory_name + str(t))
  # -------------------------------------------------------------------------------
 
  # -------------------------------- Solver ---------------------------------------
  scheme = solver.SemiImplicit_convection_diffusion1D(scheme_option)
  scheme.taylor_galerkin(c, vx, dt, M, K, G, condition_concentration.LHS, condition_concentration.bc_dirichlet, condition_concentration.bc_2)
  c = scheme.c
  # -------------------------------------------------------------------------------


# Semi Lagrangian Linear
elif scheme_option == 2:

 start_time = time()
 for t in tqdm(range(0, nt)):
 
  # ------------------------ Export VTK File --------------------------------------
  save = export_vtk.Linear1D(mesh.x,mesh.IEN,mesh.npoints,mesh.nelem,c,c,c,vx,vx)
  save.create_dir(directory_name)
  save.saveVTK(directory_name + str(t))
  # -------------------------------------------------------------------------------

  # -------------------------------- Solver ---------------------------------------
  scheme = solver.SemiImplicit_convection_diffusion1D(scheme_option)
  scheme.semi_lagrangian(mesh.npoints, mesh.neighbors_elements, mesh.IEN, mesh.x, vx, dt, c, M, condition_concentration.LHS, condition_concentration.bc_dirichlet, condition_concentration.bc_2)
  c = scheme.c
  # -------------------------------------------------------------------------------

# Semi Lagrangian Quadratic
elif scheme_option == 3:

 start_time = time()
 for t in tqdm(range(0, nt)):
 
  # ------------------------ Export VTK File --------------------------------------
  save = export_vtk.Linear1D(mesh.x,mesh.IEN,mesh.npoints,mesh.nelem,c,c,c,vx,vx)
  save.create_dir(directory_name)
  save.saveVTK(directory_name + str(t))
  # -------------------------------------------------------------------------------
 
  # -------------------------------- Solver ---------------------------------------
  scheme = solver.SemiImplicit_convection_diffusion1D(scheme_option)
  scheme.semi_lagrangian_quad(mesh.npoints, mesh.nelem, mesh.neighbors_elements, mesh.IEN, mesh.x, vx, dt, c, M, condition_concentration.LHS, condition_concentration.bc_dirichlet, condition_concentration.bc_2)
  c = scheme.c
  # -------------------------------------------------------------------------------



end_time = time()
solution_time = end_time - start_time

relatory.export(directory_name, sys.argv[0], scheme.scheme_name, mesh_name, equation_number, mesh.npoints, mesh.nelem, mesh.length_min, dt, nt, Re, Sc, import_mesh_time, assembly_time, bc_apply_time, solution_time, polynomial_order, gausspoints)
