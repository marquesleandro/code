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
import benchmark_problems
import simulator_solver
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

# ----------------------------------------------------------------------------
print ' (1) - Simulator 1D'
print ' (2) - Simulator 2D'
simulator_option = int(raw_input(" Enter simulator option above: "))
print ""
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
if simulator_option == 1:
 print ' (1) - Pure Convection'
 simulator_problem = int(raw_input(" Enter simulator problem above: "))
 print ""

elif simulator_option == 2:
 print ' (4) - Pure Convection'
 simulator_problem = int(raw_input(" Enter simulator problem above: "))
 print ""
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
mesh_name = (raw_input(" Enter name (.msh): ") + '.msh')
equation_number = int(raw_input(" Enter equation number: "))
print ""

Re = float(raw_input(" Enter Reynolds Number (Re): "))
Sc = float(raw_input(" Enter Schmidt Number (Sc): "))
print ""
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
if simulator_option == 1:
 print ' (1) - Linear Element'
 print ' (2) - Quadratic Element'
 polynomial_option = int(raw_input(" Enter polynomial degree option above: "))
 print ""

elif simulator_option == 2:
 print ' (1) - Linear Element'
 print ' (2) - Mini Element'
 print ' (3) - Quadratic Element'
 print ' (4) - Cubic Element'
 polynomial_option = int(raw_input(" Enter polynomial degree option above: "))
 print ""
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
if simulator_option == 1:
 print '  3 Gauss Points'
 print '  4 Gauss Points'
 print '  5 Gauss Points'
 print ' 10 Gauss Points'
 gausspoints = int(raw_input(" Enter Gauss Points Number option above: "))
 print ""

elif simulator_option == 2:
 print ' 3 Gauss Points'
 print ' 4 Gauss Points'
 print ' 6 Gauss Points'
 print ' 12 Gauss Points'
 gausspoints = int(raw_input(" Enter Gauss Points Number option above: "))
 print ""
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
print ' (1) - Taylor Galerkin Scheme'
print ' (2) - Semi Lagrangian Scheme'
scheme_option = int(raw_input(" Enter simulation scheme option above: "))
print ""
print ""
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
nt = int(raw_input(" Enter number of time interations (nt): "))
directory_name = raw_input(" Enter folder name to save simulations: ")
print ""
# ----------------------------------------------------------------------------




print ' ------------'
print ' IMPORT MESH:'
print ' ------------'

start_time = time()

directory = search_file.Find(mesh_name)
if directory == 'File not found':
 sys.exit()

if simulator_option == 1:
 npoints, nelem, x, IEN, neumann_pts, dirichlet_pts, neighbors_nodes, neighbors_elements, far_neighbors_nodes, far_neighbors_elements, length_min, GL, nphysical = import_msh.Element1D(directory, mesh_name, equation_number, polynomial_option)

 CFL = 0.5
 dt = float(CFL*length_min)


elif simulator_option == 2:
 npoints, nelem, x, y, IEN, neumann_edges, dirichlet_pts, neighbors_nodes, neighbors_elements, far_neighbors_nodes, far_neighbors_elements, length_min, GL, nphysical = import_msh.Element2D(directory, mesh_name, equation_number, polynomial_option)

 CFL = 0.5
 dt = float(CFL*length_min)

end_time = time()
import_mesh_time = end_time - start_time
print ' time duration: %.1f seconds' %import_mesh_time
print ""





print ' ---------'
print ' ASSEMBLY:'
print ' ---------'

start_time = time()

if simulator_option == 1:
 K, M, G, polynomial_order = assembly.Element1D(polynomial_option, GL, npoints, nelem, IEN, x, gausspoints)


elif simulator_option == 2:
 Kxx, Kxy, Kyx, Kyy, K, M, MLump, Gx, Gy, polynomial_order = assembly.Element2D(polynomial_option, GL, npoints, nelem, IEN, x, y, gausspoints)


end_time = time()
assembly_time = end_time - start_time
print ' time duration: %.1f seconds' %assembly_time
print ""





print ' --------------------------------'
print ' INITIAL AND BOUNDARY CONDITIONS:'
print ' --------------------------------'

start_time = time()

if simulator_option == 1:
 condition_concentration_LHS0 = sps.lil_matrix.copy(M)/dt
 condition_concentration = benchmark_problems.Convection1D(nphysical,npoints,x)
 condition_concentration.neumann_condition(neumann_pts[1])
 condition_concentration.dirichlet_condition(dirichlet_pts[1])
 condition_concentration.gaussian_elimination(condition_concentration_LHS0,neighbors_nodes)
 condition_concentration.initial_condition()

 LHS = condition_concentration.LHS
 bc_dirichlet = condition_concentration.bc_dirichlet
 bc_neumann = condition_concentration.bc_neumann
 bc_2 = condition_concentration.bc_2
 scalar = np.copy(condition_concentration.c)
 vx = np.copy(condition_concentration.vx)

elif simulator_option == 2:
 condition_concentration_LHS0 = sps.lil_matrix.copy(M)/dt
 condition_concentration = benchmark_problems.Convection2D(nphysical,npoints,x,y)
 condition_concentration.neumann_condition(neumann_edges[1])
 condition_concentration.dirichlet_condition(dirichlet_pts[1])
 condition_concentration.gaussian_elimination(condition_concentration_LHS0,neighbors_nodes)
 condition_concentration.initial_condition()

 LHS = condition_concentration.LHS
 bc_dirichlet = condition_concentration.bc_dirichlet
 bc_neumann = condition_concentration.bc_neumann
 bc_2 = condition_concentration.bc_2
 scalar = np.copy(condition_concentration.c)
 vx = np.copy(condition_concentration.vx)
 vy = np.copy(condition_concentration.vy)





end_time = time()
bc_apply_time = end_time - start_time
print ' time duration: %.1f seconds' %bc_apply_time
print ""





print ' -----------------------------'
print ' PARAMETERS OF THE SIMULATION:'
print ' -----------------------------'

print ' Mesh: %s' %mesh_name
print ' Number of equation: %s' %equation_number
print ' Number of nodes: %s' %npoints
print ' Number of elements: %s' %nelem
print ' Smallest edge length: %f' %length_min
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

if simulator_option == 1: #Simulator 1D
 c, scheme_name = simulator_solver.Element1D(simulator_problem, scheme_option, polynomial_option, x, IEN, npoints, nelem, scalar, vx, dt, nt, Re, Sc, M, K, G, LHS, bc_dirichlet, bc_neumann, bc_2, neighbors_nodes, neighbors_elements, directory_name)


elif simulator_option == 2: #Simulator 2D
 c, scheme_name = simulator_solver.Element2D(simulator_problem, scheme_option, polynomial_option, x, y, IEN, npoints, nelem, scalar, vx, vy, dt, nt, Re, Sc, M, Kxx, Kyx, Kxy, Kyy, Gx, Gy, LHS, bc_dirichlet, bc_neumann, bc_2, neighbors_nodes, neighbors_elements, directory_name)



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
relatory.export(directory_name, sys.argv[0], simulator_problem, scheme_name, mesh_name, equation_number, npoints, nelem, length_min, dt, nt, Re, Sc, import_mesh_time, assembly_time, bc_apply_time, solution_time, polynomial_order, gausspoints)

