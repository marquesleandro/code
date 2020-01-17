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

print ' (1) - Poiseuille'
print ' (2) - Half Poiseuille'
print ' (3) - Cavity'
print ' (4) - Pure Convection'
simulator_option = int(raw_input(" Enter simulator option above: "))
print ""


mesh_name = (raw_input(" Enter mesh name (.msh): ") + '.msh')
equation_number = int(raw_input(" Enter equation number: "))
print ""

Re = float(raw_input(" Enter Reynolds Number (Re): "))
Sc = float(raw_input(" Enter Schmidt Number (Sc): "))
print ""

print ' (1) - Linear Element'
print ' (2) - Mini Element'
print ' (3) - Quadratic Element'
print ' (4) - Cubic Element'
polynomial_option = int(raw_input(" Enter polynomial degree option above: "))
print ""

print ' 3 Gauss Points'
print ' 4 Gauss Points'
print ' 6 Gauss Points'
print ' 12 Gauss Points'
gausspoints = int(raw_input(" Enter Gauss Points Number option above: "))
print ""


nt = int(raw_input(" Enter number of time interations (nt): "))
directory_name = raw_input(" Enter folder name to save simulations: ")
print ""

print ' (1) - Taylor Galerkin'
print ' (2) - Semi Lagrangian Linear'
print ' (3) - Semi Lagrangian Mini'
print ' (4) - Semi Lagrangian Quadratic'
print ' (5) - Semi Lagrangian Cubic'
scheme_option = int(raw_input(" Enter simulation scheme option above: "))
print ""
print ""



print ' ------------'
print ' IMPORT MESH:'
print ' ------------'

start_time = time()

directory = search_file.Find(mesh_name)
if directory == 'File not found':
 sys.exit()

if polynomial_option == 1:
 mesh = import_msh.Linear2D(directory,mesh_name,equation_number)
 mesh.coord()
 mesh.ien()

elif polynomial_option == 2:
 mesh = import_msh.Mini2D(directory,mesh_name,equation_number)
 mesh.coord()
 mesh.ien()

elif polynomial_option == 3:
 mesh = import_msh.Quad2D(directory,mesh_name,equation_number)
 mesh.coord()
 mesh.ien()

elif polynomial_option == 4:
 mesh = import_msh.Cubic2D(directory,mesh_name,equation_number)
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

Kxx, Kxy, Kyx, Kyy, K, M, MLump, Gx, Gy, polynomial_order = assembly.Element2D(polynomial_option, mesh.GL, mesh.npoints, mesh.nelem, mesh.IEN, mesh.x, mesh.y, gausspoints)

LHS_c0 = (sps.lil_matrix.copy(M)/dt)

end_time = time()
assembly_time = end_time - start_time
print ' time duration: %.1f seconds' %assembly_time
print ""





print ' --------------------------------'
print ' INITIAL AND BOUNDARY CONDITIONS:'
print ' --------------------------------'

start_time = time()

if simulator_option == 4:
 # --------- Boundaries conditions --------------------
 condition_concentration = bc_apply.Convection2D(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
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

else:
 print ""
 print " Error: BC Apply not found"
 print ""
 sys.exit()


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



if simulator_option == 4: #Convection

 # Taylor Galerkin
 if scheme_option == 1:
 
  start_time = time()
  for t in tqdm(range(0, nt)):

   # ------------------------ Export VTK File ---------------------------------------
   save = export_vtk.Linear2D(mesh.x,mesh.y,mesh.IEN,mesh.npoints,mesh.nelem,c,c,c,vx,vy)
   save.create_dir(directory_name)
   save.saveVTK(directory_name + str(t))
   # --------------------------------------------------------------------------------

   # -------------------------------- Solver ---------------------------------------
   scheme = solver.SemiImplicit_concentration_equation2D(scheme_option)
   scheme.taylor_galerkin(c, vx, vy, dt, Re, Sc, M, Kxx, Kyx, Kxy, Kyy, Gx, Gy, condition_concentration.LHS, condition_concentration.bc_dirichlet, condition_concentration.bc_neumann, condition_concentration.bc_2)
   c = scheme.c
   # -------------------------------------------------------------------------------


 # Semi Lagrangian Linear
 elif scheme_option == 2:
 
  start_time = time()
  for t in tqdm(range(0, nt)):
   # ------------------------ Export VTK File ---------------------------------------
   save = export_vtk.Linear2D(mesh.x,mesh.y,mesh.IEN,mesh.npoints,mesh.nelem,c,c,c,vx,vy)
   save.create_dir(directory_name)
   save.saveVTK(directory_name + str(t))
   # --------------------------------------------------------------------------------

   # -------------------------------- Solver ---------------------------------------
   scheme = solver.SemiImplicit_concentration_equation2D(scheme_option)
   scheme.semi_lagrangian_linear(mesh.npoints, mesh.neighbors_nodes, mesh.neighbors_elements, mesh.IEN, mesh.x, mesh.y, vx, vy, dt, Re, Sc, c, M, condition_concentration.LHS, condition_concentration.bc_dirichlet, condition_concentration.bc_neumann, condition_concentration.bc_2)
   c = scheme.c
   # -------------------------------------------------------------------------------

 # Semi Lagrangian Mini
 elif scheme_option == 3:

  start_time = time()
  for t in tqdm(range(0, nt)):
 
   # ------------------------ Export VTK File ---------------------------------------
   save = export_vtk.Linear2D(mesh.x,mesh.y,mesh.IEN,mesh.npoints,mesh.nelem,c,c,c,vx,vy)
   save.create_dir(directory_name)
   save.saveVTK(directory_name + str(t))
   # --------------------------------------------------------------------------------

   # -------------------------------- Solver ---------------------------------------
   scheme = solver.SemiImplicit_concentration_equation2D(scheme_option)
   scheme.semi_lagrangian_mini(mesh.npoints, mesh.nelem, mesh.neighbors_elements, mesh.IEN, mesh.x, mesh.y, vx, vy, dt, Re, Sc, c, M, condition_concentration.LHS, condition_concentration.bc_dirichlet, condition_concentration.bc_neumann, condition_concentration.bc_2)
   c = scheme.c
   # -------------------------------------------------------------------------------


 # Semi Lagrangian Quadratic
 elif scheme_option == 4:

  start_time = time()
  for t in tqdm(range(0, nt)):
 
   # ------------------------ Export VTK File ---------------------------------------
   save = export_vtk.Linear2D(mesh.x,mesh.y,mesh.IEN,mesh.npoints,mesh.nelem,c,c,c,vx,vy)
   save.create_dir(directory_name)
   save.saveVTK(directory_name + str(t))
   # --------------------------------------------------------------------------------

   # -------------------------------- Solver ---------------------------------------
   scheme = solver.SemiImplicit_concentration_equation2D(scheme_option)
   scheme.semi_lagrangian_quad(mesh.npoints, mesh.nelem, mesh.neighbors_elements, mesh.IEN, mesh.x, mesh.y, vx, vy, dt, Re, Sc, c, M, condition_concentration.LHS, condition_concentration.bc_dirichlet, condition_concentration.bc_neumann, condition_concentration.bc_2)
   c = scheme.c
   # -------------------------------------------------------------------------------

 # Semi Lagrangian Cubic
 elif scheme_option == 5:

  start_time = time()
  for t in tqdm(range(0, nt)):
 
   # ------------------------ Export VTK File ---------------------------------------
   save = export_vtk.Linear2D(mesh.x,mesh.y,mesh.IEN,mesh.npoints,mesh.nelem,c,c,c,vx,vy)
   save.create_dir(directory_name)
   save.saveVTK(directory_name + str(t))
   # --------------------------------------------------------------------------------

   # -------------------------------- Solver ---------------------------------------
   scheme = solver.SemiImplicit_concentration_equation2D(scheme_option)
   scheme.semi_lagrangian_cubic(mesh.npoints, mesh.nelem, mesh.neighbors_elements, mesh.IEN, mesh.x, mesh.y, vx, vy, dt, Re, Sc, c, M, condition_concentration.LHS, condition_concentration.bc_dirichlet, condition_concentration.bc_neumann, condition_concentration.bc_2)
   c = scheme.c
   # -------------------------------------------------------------------------------


 else:
  print ""
  print " Error: Simulator Scheme not found"
  print ""
  sys.exit()




end_time = time()
solution_time = end_time - start_time
print ' time duration: %.1f seconds' %solution_time
print ""





print ' ----------------'
print ' SAVING RELATORY:'
print ' ----------------'
print ""
print ' End simulation. Relatory saved in %s' %directory_name
print ""

# -------------------------------- Export Relatory ---------------------------------------
relatory.export(directory_name, sys.argv[0], scheme.scheme_name, mesh_name, equation_number, mesh.npoints, mesh.nelem, mesh.length_min, dt, nt, Re, Sc, import_mesh_time, assembly_time, bc_apply_time, solution_time, polynomial_order, gausspoints)

