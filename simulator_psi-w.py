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
import semi_lagrangian
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

# ----------------------------------------------------------------------------
print ' (1) - Simulator 1D'
print ' (2) - Simulator 2D'
simulator_option = int(raw_input(" Enter simulator option above: "))
print ""
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
if simulator_option == 1:
 print ' (1) - Pure Convection'
 benchmark_problem = int(raw_input(" Enter benchmark problem above: "))
 print ""

elif simulator_option == 2:
 print ' (1) - Poiseuille'
 print ' (2) - Half Poiseuille'
 print ' (3) - Cavity'
 print ' (4) - Pure Convection'
 benchmark_problem = int(raw_input(" Enter benchmark problem above: "))
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
 #dt = 0.005

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




print '--------------------------------'
print 'INITIAL AND BOUNDARY CONDITIONS:'
print '--------------------------------'

start_time = time()


# ------------------------ Boundaries Conditions ----------------------------------
# Applying vx condition
xvelocity_LHS0 = sps.lil_matrix.copy(M)
condition_xvelocity = benchmark_problems.Half_Poiseuille(nphysical,npoints,x,y)
condition_xvelocity.neumann_condition(neumann_edges[1])
condition_xvelocity.dirichlet_condition(dirichlet_pts[1])
condition_xvelocity.gaussian_elimination(xvelocity_LHS0,neighbors_nodes)
vorticity_ibc = condition_xvelocity.ibc

# Applying vy condition
yvelocity_LHS0 = sps.lil_matrix.copy(M)
condition_yvelocity = benchmark_problems.Half_Poiseuille(nphysical,npoints,x,y)
condition_yvelocity.neumann_condition(neumann_edges[2])
condition_yvelocity.dirichlet_condition(dirichlet_pts[2])
condition_yvelocity.gaussian_elimination(yvelocity_LHS0,neighbors_nodes)

# Applying psi condition
streamfunction_LHS0 = sps.lil_matrix.copy(K)
condition_streamfunction = benchmark_problems.Half_Poiseuille(nphysical,npoints,x,y)
condition_streamfunction.streamfunction_condition(dirichlet_pts[3],streamfunction_LHS0,neighbors_nodes)
# ---------------------------------------------------------------------------------


# -------------------------- Initial condition ------------------------------------
vx = np.copy(condition_xvelocity.bc_1)
vy = np.copy(condition_yvelocity.bc_1)
psi = np.copy(condition_streamfunction.bc_1)
w = np.zeros([npoints,1], dtype = float)




#---------- Step 1 - Compute the vorticity and stream field --------------------
# -----Vorticity initial-----
vorticity_RHS = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
vorticity_LHS = sps.lil_matrix.copy(M)
w = scipy.sparse.linalg.cg(vorticity_LHS,vorticity_RHS,w, maxiter=1.0e+05, tol=1.0e-05)
w = w[0].reshape((len(w[0]),1))


# -----Streamline initial-----
# psi condition
streamfunction_RHS = sps.lil_matrix.dot(M,w)
streamfunction_RHS = np.multiply(streamfunction_RHS,condition_streamfunction.bc_2)
streamfunction_RHS = streamfunction_RHS + condition_streamfunction.bc_dirichlet
psi = scipy.sparse.linalg.cg(condition_streamfunction.LHS,streamfunction_RHS,psi, maxiter=1.0e+05, tol=1.0e-05)
psi = psi[0].reshape((len(psi[0]),1))
#----------------------------------------------------------------------------------

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



vorticity_bc_1 = np.zeros([npoints,1], dtype = float) 
for t in tqdm(range(0, nt)):
 # ------------------------ Export VTK File ---------------------------------------
 save = export_vtk.Linear2D(x,y,IEN,npoints,nelem,w,w,psi,vx,vy)
 save.create_dir(directory_name)
 save.saveVTK(directory_name + str(t))
 # --------------------------------------------------------------------------------



 #---------- Step 2 - Compute the boundary conditions for vorticity --------------
 vorticity_RHS = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
 vorticity_LHS = sps.lil_matrix.copy(M)
 vorticity_bc_1 = scipy.sparse.linalg.cg(vorticity_LHS,vorticity_RHS,vorticity_bc_1, maxiter=1.0e+05, tol=1.0e-05)
 vorticity_bc_1 = vorticity_bc_1[0].reshape((len(vorticity_bc_1[0]),1))

 
 # Gaussian elimination
 vorticity_bc_dirichlet = np.zeros([npoints,1], dtype = float)
 vorticity_bc_neumann = np.zeros([npoints,1], dtype = float)
 vorticity_bc_2 = np.ones([npoints,1], dtype = float)
 vorticity_LHS = ((np.copy(M)/dt) + (1.0/Re)*np.copy(K))
 for mm in vorticity_ibc:
  for nn in neighbors_nodes[mm]:
   vorticity_bc_dirichlet[nn] -= float(vorticity_LHS[nn,mm]*vorticity_bc_1[mm])
   vorticity_LHS[nn,mm] = 0.0
   vorticity_LHS[mm,nn] = 0.0
   
  vorticity_LHS[mm,mm] = 1.0
  vorticity_bc_dirichlet[mm] = vorticity_bc_1[mm]
  vorticity_bc_2[mm] = 0.0
 #----------------------------------------------------------------------------------



 #---------- Step 3 - Solve the vorticity transport equation ----------------------
 # Taylor Galerkin Scheme
# scheme_name = 'Taylor Galerkin'
# A = np.copy(M)/dt
# vorticity_RHS = sps.lil_matrix.dot(A,w) - np.multiply(vx,sps.lil_matrix.dot(Gx,w))\
#       - np.multiply(vy,sps.lil_matrix.dot(Gy,w))\
#       - (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(Kxx,w)) + np.multiply(vy,sps.lil_matrix.dot(Kyx,w))))\
#       - (dt/2.0)*np.multiply(vy,(np.multiply(vx,sps.lil_matrix.dot(Kxy,w)) + np.multiply(vy,sps.lil_matrix.dot(Kyy,w))))
# vorticity_RHS = np.multiply(vorticity_RHS,vorticity_bc_2)
# vorticity_RHS = vorticity_RHS + vorticity_bc_dirichlet
# w = scipy.sparse.linalg.cg(vorticity_LHS,vorticity_RHS,w, maxiter=1.0e+05, tol=1.0e-05)
# w = w[0].reshape((len(w[0]),1))


 # Semi-Lagrangian Scheme
 scheme_name = 'Semi Lagrangian'
 w_d = semi_lagrangian.Linear2D(npoints, neighbors_elements, IEN, x, y, vx, vy, dt, w)
 A = np.copy(M)/dt
 vorticity_RHS = sps.lil_matrix.dot(A,w_d)

 vorticity_RHS = vorticity_RHS + (1.0/Re)*vorticity_bc_neumann
 vorticity_RHS = np.multiply(vorticity_RHS,vorticity_bc_2)
 vorticity_RHS = vorticity_RHS + vorticity_bc_dirichlet
 
 w = scipy.sparse.linalg.cg(vorticity_LHS,vorticity_RHS,w, maxiter=1.0e+05, tol=1.0e-05)
 w = w[0].reshape((len(w[0]),1))
 #----------------------------------------------------------------------------------



 #---------- Step 4 - Solve the streamline equation --------------------------------
 # Solve Streamline
 # psi condition
 streamfunction_RHS = sps.lil_matrix.dot(M,w)
 streamfunction_RHS = np.multiply(streamfunction_RHS,condition_streamfunction.bc_2)
 streamfunction_RHS = streamfunction_RHS + condition_streamfunction.bc_dirichlet
 psi = scipy.sparse.linalg.cg(condition_streamfunction.LHS,streamfunction_RHS,psi, maxiter=1.0e+05, tol=1.0e-05)
 psi = psi[0].reshape((len(psi[0]),1))
 #----------------------------------------------------------------------------------



 #---------- Step 5 - Compute the velocity field -----------------------------------
 # Velocity vx
 xvelocity_RHS = sps.lil_matrix.dot(Gy,psi)
 xvelocity_RHS = np.multiply(xvelocity_RHS,condition_xvelocity.bc_2)
 xvelocity_RHS = xvelocity_RHS + condition_xvelocity.bc_dirichlet
 vx = scipy.sparse.linalg.cg(condition_xvelocity.LHS,xvelocity_RHS,vx, maxiter=1.0e+05, tol=1.0e-05)
 vx = vx[0].reshape((len(vx[0]),1))
 
 # Velocity vy
 yvelocity_RHS = -sps.lil_matrix.dot(Gx,psi)
 yvelocity_RHS = np.multiply(yvelocity_RHS,condition_yvelocity.bc_2)
 yvelocity_RHS = yvelocity_RHS + condition_yvelocity.bc_dirichlet
 vy = scipy.sparse.linalg.cg(condition_yvelocity.LHS,yvelocity_RHS,vy, maxiter=1.0e+05, tol=1.0e-05)
 vy = vy[0].reshape((len(vy[0]),1))
 #----------------------------------------------------------------------------------


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
relatory.export(directory_name, sys.argv[0], benchmark_problem, scheme_name, mesh_name, equation_number, npoints, nelem, length_min, dt, nt, Re, Sc, import_mesh_time, assembly_time, bc_apply_time, solution_time, polynomial_order, gausspoints)



