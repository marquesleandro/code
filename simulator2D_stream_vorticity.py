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
 Simulator created by Leandro Marques at 12/2018
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
mesh_name = raw_input(" Enter mesh name: ")
equation_number = int(raw_input(" Enter equation number: "))
nt = int(raw_input(" Enter number of time interations (nt): "))
Re = float(raw_input(" Enter Reynolds Number (Re): "))
Sc = float(raw_input(" Enter Schmidt Number (Sc): "))
directory_name = raw_input(" Enter folder name to save simulations: ")
file_simulation = (sys.argv[0].split('.py'))[0]
print ""




print ' ------------'
print ' IMPORT MESH:'
print ' ------------'

start_time = time()

directory = search_file.Find(mesh_name)
if directory == 'File does not found':
 sys.exit()

mesh = import_msh.Linear2D(directory,mesh_name,equation_number)
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

Kxx, Kxy, Kyx, Kyy, K, M, MLump, Gx, Gy = assembly.Linear2D(mesh.GL, mesh.npoints, mesh.nelem, mesh.IEN, mesh.x, mesh.y)

LHS_vx0 = sps.lil_matrix.copy(M)
LHS_vy0 = sps.lil_matrix.copy(M)
LHS_psi0 = sps.lil_matrix.copy(K)
#LHS_c0 = ((sps.lil_matrix.copy(M)/dt) + (1.0/(Re*Sc))*sps.lil_matrix.copy(K))

end_time = time()
print ' time duration: %.1f seconds' %(end_time - start_time)
print ""




print ' --------------------------------'
print ' INITIAL AND BOUNDARY CONDITIONS:'
print ' --------------------------------'

start_time = time()


# ------------------------ Boundaries Conditions ----------------------------------
# Applying vx condition
condition_xvelocity = bc_apply.Cavity(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
condition_xvelocity.neumann_condition(mesh.neumann_edges[1])
condition_xvelocity.dirichlet_condition(mesh.dirichlet_pts[1])
condition_xvelocity.gaussian_elimination(LHS_vx0,mesh.neighbors_nodes)
vorticity_ibc = condition_xvelocity.ibc

# Applying vy condition
condition_yvelocity = bc_apply.Cavity(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
condition_yvelocity.neumann_condition(mesh.neumann_edges[2])
condition_yvelocity.dirichlet_condition(mesh.dirichlet_pts[2])
condition_yvelocity.gaussian_elimination(LHS_vy0,mesh.neighbors_nodes)

# Applying psi condition
#condition_streamfunction = bc_apply.Cavity(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
#condition_streamfunction.streamfunction_condition(mesh.dirichlet_pts[3],LHS_psi0,mesh.neighbors_nodes)

condition_streamfunction = bc_apply.Cavity(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
condition_streamfunction.neumann_condition(mesh.neumann_edges[3])
condition_streamfunction.dirichlet_condition(mesh.dirichlet_pts[3])
condition_streamfunction.gaussian_elimination(LHS_psi0,mesh.neighbors_nodes)

# Applying concentration condition
#condition_concentration = bc_apply.Cavity(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
#condition_concentration.neumann_condition(mesh.neumann_edges[4])
#condition_concentration.dirichlet_condition(mesh.dirichlet_pts[4])
#condition_concentration.gaussian_elimination(LHS_c0,mesh.neighbors_nodes)
# ---------------------------------------------------------------------------------


# -------------------------- Initial condition ------------------------------------
vx = np.copy(condition_xvelocity.bc_1)
vy = np.copy(condition_yvelocity.bc_1)
psi = np.copy(condition_streamfunction.bc_1)
#c = np.copy(condition_concentration.bc_1)
w = np.zeros([mesh.npoints,1], dtype = float)


# Step 1 - Vorticity initial
vorticity_RHS = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
vorticity_LHS = sps.lil_matrix.copy(M)

w = scipy.sparse.linalg.cg(vorticity_LHS,vorticity_RHS,w, maxiter=1.0e+05, tol=1.0e-05)
w = w[0].reshape((len(w[0]),1))


# Step 1 - Streamline initial
streamfunction_RHS = sps.lil_matrix.dot(M,w)
streamfunction_RHS = np.multiply(streamfunction_RHS,condition_streamfunction.bc_2)
streamfunction_RHS = streamfunction_RHS + condition_streamfunction.bc_dirichlet

psi = scipy.sparse.linalg.cg(condition_streamfunction.LHS,streamfunction_RHS,psi, maxiter=1.0e+05, tol=1.0e-05)
psi = psi[0].reshape((len(psi[0]),1))
# ---------------------------------------------------------------------------------

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

vorticity_bc_1 = np.zeros([mesh.npoints,1], dtype = float) 

for t in tqdm(range(0, nt)):

 # ------------------------ Export VTK File ---------------------------------------
 save = export_vtk.Linear2D(mesh.x,mesh.y,mesh.IEN,mesh.npoints,mesh.nelem,w,w,psi,vx,vy)
 save.create_dir(directory_name)
 save.saveVTK(file_simulation + str(t))
 # --------------------------------------------------------------------------------


 # ---------- Step 2 - Compute the boundary conditions for vorticity --------------
 vorticity_RHS = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
 vorticity_LHS = sps.lil_matrix.copy(M)

 vorticity_bc_1 = scipy.sparse.linalg.cg(M,vorticity_RHS,vorticity_bc_1, maxiter=1.0e+05, tol=1.0e-05)
 vorticity_bc_1 = vorticity_bc_1[0].reshape((len(vorticity_bc_1[0]),1))


 # Gaussian elimination
 vorticity_bc_dirichlet = np.zeros([mesh.npoints,1], dtype = float)
 vorticity_bc_neumann = np.zeros([mesh.npoints,1], dtype = float)
 vorticity_bc_2 = np.ones([mesh.npoints,1], dtype = float)
 
 vorticity_LHS = ((np.copy(M)/dt) + (1.0/Re)*np.copy(K))
 for mm in vorticity_ibc:
  for nn in mesh.neighbors_nodes[mm]:
   vorticity_bc_dirichlet[nn] -= float(vorticity_LHS[nn,mm]*vorticity_bc_1[mm])
   vorticity_LHS[nn,mm] = 0.0
   vorticity_LHS[mm,nn] = 0.0
   
  vorticity_LHS[mm,mm] = 1.0
  vorticity_bc_dirichlet[mm] = vorticity_bc_1[mm]
  vorticity_bc_2[mm] = 0.0
 # --------------------------------------------------------------------------------


 # --------- Step 3 - Solve the vorticity transport equation ----------------------
 # Taylor Galerkin Scheme
# A = np.copy(M)/dt
# vorticity_RHS = sps.lil_matrix.dot(A,w) - np.multiply(vx,sps.lil_matrix.dot(Gx,w))\
#                                         - np.multiply(vy,sps.lil_matrix.dot(Gy,w))\
#                - (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(Kxx,w))\
#                                         + np.multiply(vy,sps.lil_matrix.dot(Kyx,w))))\
#                - (dt/2.0)*np.multiply(vy,(np.multiply(vx,sps.lil_matrix.dot(Kxy,w))\
#                                         + np.multiply(vy,sps.lil_matrix.dot(Kyy,w))))
#
# vorticity_RHS = vorticity_RHS + (1.0/Re)*vorticity_bc_neumann
# vorticity_RHS = np.multiply(vorticity_RHS,vorticity_bc_2)
# vorticity_RHS = vorticity_RHS + vorticity_bc_dirichlet
# 
# w = scipy.sparse.linalg.cg(vorticity_LHS,vorticity_RHS,w, maxiter=1.0e+05, tol=1.0e-05)
# w = w[0].reshape((len(w[0]),1))


 # Semi-Lagrangian Scheme
 x_d = mesh.x - vx*dt
 y_d = mesh.y - vy*dt

 w_d = semi_lagrangian.Linear2D(mesh.npoints, mesh.IEN, mesh.x, mesh.y, x_d, y_d, mesh.neighbors_elements, w)

 A = np.copy(M)/dt
 vorticity_RHS = sps.lil_matrix.dot(A,w_d)

 vorticity_RHS = vorticity_RHS + (1.0/Re)*vorticity_bc_neumann
 vorticity_RHS = np.multiply(vorticity_RHS,vorticity_bc_2)
 vorticity_RHS = vorticity_RHS + vorticity_bc_dirichlet
 
 w = scipy.sparse.linalg.cg(vorticity_LHS,vorticity_RHS,w, maxiter=1.0e+05, tol=1.0e-05)
 w = w[0].reshape((len(w[0]),1))
 # --------------------------------------------------------------------------------



 # -------- Step 4 - Solve the streamline equation --------------------------------
 streamfunction_RHS = sps.lil_matrix.dot(M,w)
 streamfunction_RHS = np.multiply(streamfunction_RHS,condition_streamfunction.bc_2)
 streamfunction_RHS = streamfunction_RHS + condition_streamfunction.bc_dirichlet
 psi = scipy.sparse.linalg.cg(condition_streamfunction.LHS,streamfunction_RHS,psi, maxiter=1.0e+05, tol=1.0e-05)
 psi = psi[0].reshape((len(psi[0]),1))
 # --------------------------------------------------------------------------------



 # -------- Step 5 - Compute the velocity field -----------------------------------
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
 # --------------------------------------------------------------------------------



 # ---------- Step 6 - Concentration ----------------------------------------------
# A = np.copy(M)/dt
# concentration_RHS = sps.lil_matrix.dot(A,c) - np.multiply(vx,sps.lil_matrix.dot(Gx,c))\
#                                             - np.multiply(vy,sps.lil_matrix.dot(Gy,c))\
#                    - (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(Kxx,c))\
#                                             + np.multiply(vy,sps.lil_matrix.dot(Kyx,c))))\
#                    - (dt/2.0)*np.multiply(vy,(np.multiply(vx,sps.lil_matrix.dot(Kxy,c))\
#                                             + np.multiply(vy,sps.lil_matrix.dot(Kyy,c))))
# 
# concentration_RHS = concentration_RHS + (1.0/(Re*Sc))*condition_concentration.bc_neumann
# concentration_RHS = np.multiply(concentration_RHS,condition_concentration.bc_2)
# concentration_RHS = concentration_RHS + condition_concentration.bc_dirichlet
# 
# c = scipy.sparse.linalg.cg(condition_concentration.LHS,concentration_RHS,c, maxiter=1.0e+05, tol=1.0e-05)
# c = c[0].reshape((len(c[0]),1))
 # --------------------------------------------------------------------------------
