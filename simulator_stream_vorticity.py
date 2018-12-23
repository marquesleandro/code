# ==========================================
# Code created by Leandro Marques at 12/2018
# Gesar Search Group
# State University of the Rio de Janeiro
# e-mail: marquesleandro67@gmail.com
# ==========================================

# FEM Simulator - Streamfunction-Vorticity Formulation



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
import export_vtk


print '------------'
print 'IMPORT MESH:'
print '------------'

start_time = time()

name_mesh = 'malha_poiseuille.msh'
number_equation = 3
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
dt = 0.0005
nt = 20000
theta = 1.0

# Nondimensional Numbers
Re = 54.5
Sc = 1.0

# --------------------- Parameters of the Simulation ------------------------------
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
# ---------------------------------------------------------------------------------



print '--------'
print 'ASSEMBLY:'
print '--------'

start_time = time()

Kxx, Kxy, Kyx, Kyy, K, M, MLump, Gx, Gy = assembly.Linear(mesh.npoints, mesh.nelem, mesh.IEN, mesh.x, mesh.y)


LHS_vx0 = sps.lil_matrix.copy(M)
LHS_vy0 = sps.lil_matrix.copy(M)
LHS_psi0 = sps.lil_matrix.copy(K)
#LHS_c0 = ((sps.lil_matrix.copy(M)/dt) + (theta/(Re*Sc))*sps.lil_matrix.copy(K))


end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""



print '--------------------------------'
print 'INITIAL AND BOUNDARY CONDITIONS:'
print '--------------------------------'

start_time = time()

# ------------------------ Boundaries Conditions ----------------------------------

# Applying vx condition
condition_xvelocity = bc_apply.Poiseuille(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
condition_xvelocity.neumann_condition(mesh.neumann_edges[1])
condition_xvelocity.dirichlet_condition(mesh.dirichlet_pts[1])
condition_xvelocity.gaussian_elimination(LHS_vx0,mesh.neighbors_nodes)
vorticity_ibc = condition_xvelocity.ibc

# Applying vy condition
condition_yvelocity = bc_apply.Poiseuille(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
condition_yvelocity.neumann_condition(mesh.neumann_edges[2])
condition_yvelocity.dirichlet_condition(mesh.dirichlet_pts[2])
condition_yvelocity.gaussian_elimination(LHS_vy0,mesh.neighbors_nodes)

# Applying psi condition
condition_streamfunction = bc_apply.Poiseuille(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
condition_streamfunction.streamfunction_condition(mesh.dirichlet_pts[3],LHS_psi0,mesh.neighbors_nodes)

# Applying concentration condition
#condition_concentration = bc_apply.Linear(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
#condition_concentration.neumann_condition(mesh.neumann_edges[4])
#condition_concentration.dirichlet_condition(mesh.dirichlet_pts[4])
#condition_concentration.gaussian_elimination(LHS_c0,mesh.neighbors_nodes)

# -------------------------- Initial condition ------------------------------------
vx = np.copy(condition_xvelocity.bc_1)
vy = np.copy(condition_yvelocity.bc_1)
psi = np.copy(condition_streamfunction.bc_1)
#c = np.copy(condition_concentration.bc_1)
w = np.zeros([mesh.npoints,1], dtype = float)


# ------------ Step 1 - Compute the vorticity and stream field --------------------
# Vorticity initial
AA = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
w = scipy.sparse.linalg.cg(M,AA,w, maxiter=1.0e+05, tol=1.0e-05)
w = w[0].reshape((len(w[0]),1))


# Streamline initial
vorticity_RHS = sps.lil_matrix.dot(M,w)
vorticity_RHS = np.multiply(vorticity_RHS,condition_streamfunction.bc_2)
vorticity_RHS += condition_streamfunction.bc_dirichlet
psi = scipy.sparse.linalg.cg(condition_streamfunction.LHS,vorticity_RHS,psi, maxiter=1.0e+05, tol=1.0e-05)
psi = psi[0].reshape((len(psi[0]),1))
# ---------------------------------------------------------------------------------


end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""



print '----------------------------'
print 'SOLVE THE LINEARS EQUATIONS:'
print '----------------------------'
print ""


vorticity_bc_1 = np.zeros([mesh.npoints,1], dtype = float) 

for t in tqdm(range(0, nt)):

 # ------------------------ Export VTK File ---------------------------------------
# save = export_vtk.Linear(mesh.x,mesh.y,mesh.IEN,mesh.npoints,mesh.nelem,c,w,psi,vx,vy)
# save.saveVTK('/home/marquesleandro/results/result_coronary_RealGeoStrut_ext1','coronary%s' %t)
 # --------------------------------------------------------------------------------


 # ---------- Step 2 - Compute the boundary conditions for vorticity --------------
 AA = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
 vorticity_bc_1 = scipy.sparse.linalg.cg(M,AA,vorticity_bc_1, maxiter=1.0e+05, tol=1.0e-05)
 vorticity_bc_1 = vorticity_bc_1[0].reshape((len(vorticity_bc_1[0]),1))

 # Gaussian elimination
 vorticity_bc_dirichlet = np.zeros([mesh.npoints,1], dtype = float)
 vorticity_bc_neumann = np.zeros([mesh.npoints,1], dtype = float)
 vorticity_bc_2 = np.ones([mesh.npoints,1], dtype = float)
 
 vorticity_LHS = ((np.copy(M)/dt) + (theta/Re)*np.copy(K))
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
 A = np.copy(M)/dt
 vorticity_RHS = sps.lil_matrix.dot(A,w) - np.multiply(vx,sps.lil_matrix.dot(Gx,w))\
                                         - np.multiply(vy,sps.lil_matrix.dot(Gy,w))\
                - (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(Kxx,w))\
                                         + np.multiply(vy,sps.lil_matrix.dot(Kyx,w))))\
                - (dt/2.0)*np.multiply(vy,(np.multiply(vx,sps.lil_matrix.dot(Kxy,w))\
                                         + np.multiply(vy,sps.lil_matrix.dot(Kyy,w))))

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
 AA = sps.lil_matrix.dot(Gy,psi)
 xvelocity_RHS = np.multiply(AA,condition_xvelocity.bc_2)
 xvelocity_RHS = xvelocity_RHS + condition_xvelocity.bc_dirichlet
 vx = scipy.sparse.linalg.cg(condition_xvelocity.LHS,xvelocity_RHS,vx, maxiter=1.0e+05, tol=1.0e-05)
 vx = vx[0].reshape((len(vx[0]),1))
 
 # Velocity vy
 AA = -sps.lil_matrix.dot(Gx,psi)
 yvelocity_RHS = np.multiply(AA,condition_yvelocity.bc_2)
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
