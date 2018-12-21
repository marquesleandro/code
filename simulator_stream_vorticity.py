# FEM Simulator - Streamfunction-Vorticity Formulation
# Created by Leandro Marques
# 14.11.18 


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
bc = np.zeros([mesh.nphysical,1], dtype = float) 

# Velocity vx condition
bc[0][0] = 0.0
bc[1][0] = 0.0
bc[2][0] = 1.0
bc[3][0] = 0.0

# Velocity vy condition
bc[4][0] = 0.0
bc[5][0] = 0.0
bc[6][0] = 0.0
bc[7][0] = 0.0

# Concentration condition
#bc[12][0] = 0.0
#bc[13][0] = 0.0
#bc[14][0] = 0.0
#bc[15][0] = 0.0
#bc[16][0] = 1.0

# Applying vx condition
condition_xvelocity = bc_apply.Poiseuille(mesh.npoints,mesh.x,mesh.y,bc)
condition_xvelocity.neumann_condition(mesh.neumann_edges[1])
condition_xvelocity.dirichlet_condition(mesh.dirichlet_pts[1])
condition_xvelocity.gaussian_elimination(LHS_vx0,mesh.neighbors_nodes)
ibc_w = condition_xvelocity.ibc

# Applying vy condition
condition_yvelocity = bc_apply.Poiseuille(mesh.npoints,mesh.x,mesh.y,bc)
condition_yvelocity.neumann_condition(mesh.neumann_edges[2])
condition_yvelocity.dirichlet_condition(mesh.dirichlet_pts[2])
condition_yvelocity.gaussian_elimination(LHS_vy0,mesh.neighbors_nodes)

# Applying psi condition
condition_psi = bc_apply.Poiseuille(mesh.npoints,mesh.x,mesh.y,bc)
condition_psi.streamfunction_condition(mesh.dirichlet_pts[3],LHS_psi0,mesh.neighbors_nodes)

# Applying concentration condition
#condition_concentration = bc_apply.Linear(mesh.npoints,mesh.x,mesh.y,bc)
#condition_concentration.neumann_condition(mesh.neumann_edges[4])
#condition_concentration.dirichlet_condition(mesh.dirichlet_pts[4])
#condition_concentration.gaussian_elimination(LHS_c0,mesh.neighbors_nodes)

'''
# -------------------------- Initial condition ------------------------------------
vx = condition_xvelocity.bc_1
vy = condition_yvelocity.bc_1
psi = bc_1_psi
c = condition_concentration.bc_1
w = np.zeros([mesh.npoints,1], dtype = float)


# ------------ Step 1 - Compute the vorticity and stream field --------------------
# Vorticity initial
AA = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
w = scipy.sparse.linalg.cg(M,AA,w, maxiter=1.0e+05, tol=1.0e-05)
w = w[0].reshape((len(w[0]),1))


# Streamline initial
RHS_psi = sps.lil_matrix.dot(M,w)
RHS_psi = np.multiply(RHS_psi,bc_2_psi)
RHS_psi += bc_dirichlet_psi
psi = scipy.sparse.linalg.cg(LHS_psi,RHS_psi,psi, maxiter=1.0e+05, tol=1.0e-05)
psi = psi[0].reshape((len(psi[0]),1))
# ---------------------------------------------------------------------------------


end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""



print '----------------------------'
print 'SOLVE THE LINEARS EQUATIONS:'
print '----------------------------'
print ""


bc_1_w = np.zeros([mesh.npoints,1], dtype = float) 

for t in tqdm(range(0, nt)):

 # ------------------------ Export VTK File ---------------------------------------
# save = export_vtk.Linear(mesh.x,mesh.y,mesh.IEN,mesh.npoints,mesh.nelem,c,w,psi,vx,vy)
# save.saveVTK('/home/marquesleandro/results/result_coronary_RealGeoStrut_ext1','coronary%s' %t)
 # --------------------------------------------------------------------------------


 # ---------- Step 2 - Compute the boundary conditions for vorticity --------------
 AA = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
 bc_1_w = scipy.sparse.linalg.cg(M,AA,bc_1_w, maxiter=1.0e+05, tol=1.0e-05)
 bc_1_w = bc_1_w[0].reshape((len(bc_1_w[0]),1))

 # Gaussian elimination
 bc_dirichlet_w = np.zeros([mesh.npoints,1], dtype = float)
 bc_neumann_w = np.zeros([mesh.npoints,1], dtype = float)
 bc_2_w = np.ones([mesh.npoints,1], dtype = float)
 
 LHS_w = ((np.copy(M)/dt) + (theta/Re)*np.copy(K))
 for i in range(0,len(ibc_w)):
  mm = ibc_w[i]
  for nn in mesh.neighbors_nodes[mm]:
   bc_dirichlet_w[nn] -= float(LHS_w[nn,mm]*bc_1_w[mm])
   LHS_w[nn,mm] = 0.0
   LHS_w[mm,nn] = 0.0
   
  LHS_w[mm,mm] = 1.0
  bc_dirichlet_w[mm] = bc_1_w[mm]
  bc_2_w[mm] = 0.0
 # --------------------------------------------------------------------------------



 # --------- Step 3 - Solve the vorticity transport equation ----------------------
 A = np.copy(M)/dt
 RHS_w = sps.lil_matrix.dot(A,w) - np.multiply(vx,sps.lil_matrix.dot(Gx,w))\
                                 - np.multiply(vy,sps.lil_matrix.dot(Gy,w))\
       - (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(Kxx,w))\
                                + np.multiply(vy,sps.lil_matrix.dot(Kyx,w))))\
       - (dt/2.0)*np.multiply(vy,(np.multiply(vx,sps.lil_matrix.dot(Kxy,w))\
                                + np.multiply(vy,sps.lil_matrix.dot(Kyy,w))))

 RHS_w = RHS_w + (1.0/Re)*bc_neumann_w
 RHS_w = np.multiply(RHS_w,bc_2_w)
 RHS_w = RHS_w + bc_dirichlet_w
 
 w = scipy.sparse.linalg.cg(LHS_w,RHS_w,w, maxiter=1.0e+05, tol=1.0e-05)
 w = w[0].reshape((len(w[0]),1))
 # --------------------------------------------------------------------------------



 # -------- Step 4 - Solve the streamline equation --------------------------------
 RHS_psi = sps.lil_matrix.dot(M,w)
 RHS_psi = np.multiply(RHS_psi,bc_2_psi)
 RHS_psi = RHS_psi + bc_dirichlet_psi
 psi = scipy.sparse.linalg.cg(LHS_psi,RHS_psi,psi, maxiter=1.0e+05, tol=1.0e-05)
 psi = psi[0].reshape((len(psi[0]),1))
 # --------------------------------------------------------------------------------



 # -------- Step 5 - Compute the velocity field -----------------------------------
 # Velocity vx
 AA = sps.lil_matrix.dot(Gy,psi)
 RHS_vx = np.multiply(AA,condition_xvelocity.bc_2)
 RHS_vx = RHS_vx + condition_xvelocity.bc_dirichlet
 vx = scipy.sparse.linalg.cg(condition_xvelocity.LHS,RHS_vx,vx, maxiter=1.0e+05, tol=1.0e-05)
 vx = vx[0].reshape((len(vx[0]),1))
 
 # Velocity vy
 AA = -sps.lil_matrix.dot(Gx,psi)
 RHS_vy = np.multiply(AA,condition_yvelocity.bc_2)
 RHS_vy = RHS_vy + condition_yvelocity.bc_dirichlet
 vy = scipy.sparse.linalg.cg(condition_yvelocity.LHS,RHS_vy,vy, maxiter=1.0e+05, tol=1.0e-05)
 vy = vy[0].reshape((len(vy[0]),1))
 # --------------------------------------------------------------------------------



 # ---------- Step 6 - Concentration ----------------------------------------------
 A = np.copy(M)/dt
 RHS_c = sps.lil_matrix.dot(A,c) - np.multiply(vx,sps.lil_matrix.dot(Gx,c))\
                                 - np.multiply(vy,sps.lil_matrix.dot(Gy,c))\
       - (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(Kxx,c))\
                                + np.multiply(vy,sps.lil_matrix.dot(Kyx,c))))\
       - (dt/2.0)*np.multiply(vy,(np.multiply(vx,sps.lil_matrix.dot(Kxy,c))\
                                + np.multiply(vy,sps.lil_matrix.dot(Kyy,c))))
 
 RHS_c = RHS_c + (1.0/(Re*Sc))*condition_concentration.bc_neumann
 RHS_c = np.multiply(RHS_c,condition_concentration.bc_2)
 RHS_c = RHS_c + condition_concentration.bc_dirichlet
 
 c = scipy.sparse.linalg.cg(condition_concentration.LHS,RHS_c,c, maxiter=1.0e+05, tol=1.0e-05)
 c = c[0].reshape((len(c[0]),1))
 # --------------------------------------------------------------------------------
'''
