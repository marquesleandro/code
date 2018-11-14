# ---------------- SIMULATOR ------------------
print '------------------------------'
print 'INFORMATIONS OF THE SIMULATOR:'
print '------------------------------'
print 'Simulator: Backward Facing Step Problem'
print 'Created by: Leandro Marques'
print 'Date: 14.11.18'
print ""
# -----------------------------------------------


# =======================
# Importing the libraries
# =======================

import sys
sys.path.insert(0, '/home/marquesleandro/codigook/lib_class')

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import scipy.linalg
import trimsh
import trielem
from tricond import b_bc
import InOut
from tqdm import tqdm
from time import time


print '------------'
print 'IMPORT MESH:'
print '------------'

start_time = time()

name_mesh = 'malha_backward.msh'
mesh = trimsh.Linear('../mesh',name_mesh)
mesh.parameters(3)
mesh.ien()
mesh.coord()

end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""



# ==========
# Parameters
# ==========

# Time
dt = 0.005
nt = 200 
theta = 1.0

# Nondimensional Numbers
Re = 10.0
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
print 'Time scheme (theta): %s' %theta
print 'Reynolds number: %s' %Re
print 'Schmidt number: %s' %Sc
print ""
# -----------------------------------------------


print '--------'
print 'ASSEMBLY:'
print '--------'

start_time = time()

Kxx = sps.lil_matrix((mesh.npoints,mesh.npoints), dtype = float)
Kxy = sps.lil_matrix((mesh.npoints,mesh.npoints), dtype = float)
Kyx = sps.lil_matrix((mesh.npoints,mesh.npoints), dtype = float)
Kyy = sps.lil_matrix((mesh.npoints,mesh.npoints), dtype = float)
K = sps.lil_matrix((mesh.npoints,mesh.npoints), dtype = float)
M = sps.lil_matrix((mesh.npoints,mesh.npoints), dtype = float)
MLump = sps.lil_matrix((mesh.npoints,mesh.npoints), dtype = float)
Gx = sps.lil_matrix((mesh.npoints,mesh.npoints), dtype = float)
Gy = sps.lil_matrix((mesh.npoints,mesh.npoints), dtype = float)


linear = trielem.Linear(mesh.x, mesh.y, mesh.IEN)

for e in range(0,mesh.nelem): 
# linear.analytic(e)
 linear.numerical(e)

 for i in range(0,3): 
  ii = mesh.IEN[e][i]
  
  for j in range(0,3):
   jj = mesh.IEN[e][j]

   Kxx[ii,jj] += linear.kxx[i][j]
   Kxy[ii,jj] += linear.kxy[i][j]
   Kyx[ii,jj] += linear.kyx[i][j]
   Kyy[ii,jj] += linear.kyy[i][j]
   K[ii,jj] += linear.kxx[i][j] + linear.kyy[i][j]
   M[ii,jj] += linear.mass[i][j]
   MLump[ii,ii] += linear.mass[i][j]

  for k in range(0,3):
   kk = mesh.IEN[e][k]

   Gx[ii,kk] += linear.gx[i][k]
   Gy[ii,kk] += linear.gy[i][k]


Minv = np.linalg.inv(M.todense())
MinvLump = np.linalg.inv(MLump.todense())

LHS_vx0 = sps.lil_matrix.copy(M)
LHS_vy0 = sps.lil_matrix.copy(M)
LHS_psi0 = sps.lil_matrix.copy(K)


end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""


print '--------------------------------'
print 'INITIAL AND BOUNDARY CONDITIONS:'
print '--------------------------------'

start_time = time()

# ---------Boundaries conditions--------------------
bc = np.zeros([mesh.nphysical,1], dtype = float) 

# Velocity vx condition - pg. 7
bc[0][0] = 0.0 
bc[1][0] = 1.0
bc[2][0] = 0.0

# Velocity vy condition - pg. 7
bc[3][0] = 0.0
bc[4][0] = 0.0
bc[5][0] = 0.0

# Streamline condition - pg. 7
bc[6][0] = 0.0
bc[7][0] = 0.0
bc[8][0] = 0.0
bc[9][0] = 2.0

# Applying vx condition
bc_neumann_vx,bc_dirichlet_vx,LHS_vx,bc_1_vx,bc_2_vx,ibc_vx = b_bc(mesh.npoints, mesh.x, mesh.y, bc, mesh.neumann_edges[1], mesh.dirichlet_pts[1], LHS_vx0, mesh.neighbors_nodes)
ibc_w = ibc_vx

# Applying vy condition
bc_neumann_vy,bc_dirichlet_vy,LHS_vy,bc_1_vy,bc_2_vy,ibc_vy = b_bc(mesh.npoints, mesh.x, mesh.y, bc, mesh.neumann_edges[2], mesh.dirichlet_pts[2], LHS_vy0, mesh.neighbors_nodes)

# Applying psi condition
bc_neumann_psi,bc_dirichlet_psi,LHS_psi,bc_1_psi,bc_2_psi,ibc_psi = b_bc(mesh.npoints, mesh.x, mesh.y, bc, mesh.neumann_edges[3], mesh.dirichlet_pts[3], LHS_psi0, mesh.neighbors_nodes)
#----------------------------------------------------------------------------------


# ---------Initial condition--------------------
psi = bc_dirichlet_psi
vx = bc_dirichlet_vx
vy = bc_dirichlet_vy

#---------- Step 1 - Compute the vorticity and stream field --------------------
# -----Vorticity initial-----
AA = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
w = np.dot(MinvLump,AA)

# -----Streamline initial-----
# psi condition
RHS_psi = sps.lil_matrix.dot(M,w)
RHS_psi = np.multiply(RHS_psi,bc_2_psi)
RHS_psi += bc_1_psi
psi = scipy.sparse.linalg.cg(LHS_psi,RHS_psi,psi, maxiter=1.0e+05, tol=1.0e-05)
psi = psi[0].reshape((len(psi[0]),1))
#----------------------------------------------------------------------------------

end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""



print '----------------------------'
print 'SOLVE THE LINEARS EQUATIONS:'
print '----------------------------'
print ""


for t in tqdm(range(0, nt)):
 save = InOut.Linear(mesh.x,mesh.y,mesh.IEN,mesh.npoints,mesh.nelem,w,psi,vx,vy)
 save.saveVTK('./result_backward','backward%s' %t)

 #---------- Step 2 - Compute the boundary conditions for vorticity --------------
 AA = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
 bc_dirichlet_w = np.dot(MinvLump,AA)

 # Gaussian elimination
 bc_1_w = np.zeros([mesh.npoints,1], dtype = float)
 bc_2_w = np.ones([mesh.npoints,1], dtype = float)
 LHS_w = ((np.copy(M)/dt) + (theta/Re)*np.copy(K))
 for i in range(0,len(ibc_w)):
  mm = ibc_w[i]
  for nn in mesh.neighbors_nodes[mm]:
   bc_1_w[nn] -= float(LHS_w[nn,mm]*bc_dirichlet_w[mm])
   LHS_w[nn,mm] = 0.0
   LHS_w[mm,nn] = 0.0
   
  LHS_w[mm,mm] = 1.0
  bc_1_w[mm] = bc_dirichlet_w[mm]
  bc_2_w[mm] = 0.0
 #----------------------------------------------------------------------------------

 #---------- Step 3 - Solve the vorticity transport equation ----------------------
 # Solve Vorticity
 A = np.copy(M)/dt
 RHS_w = sps.lil_matrix.dot(A,w) - np.multiply(vx,sps.lil_matrix.dot(Gx,w))\
       - np.multiply(vy,sps.lil_matrix.dot(Gy,w))\
#       - (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(Kxx,w)) + np.multiply(vy,sps.lil_matrix.dot(Kxy,w))))\
#       - (dt/2.0)*np.multiply(vy,(np.multiply(vx,sps.lil_matrix.dot(Kyx,w)) + np.multiply(vy,sps.lil_matrix.dot(Kyy,w))))
 RHS_w = np.multiply(RHS_w,bc_2_w)
 RHS_w += bc_1_w
 w = scipy.sparse.linalg.cg(LHS_w,RHS_w,w, maxiter=1.0e+05, tol=1.0e-05)
 w = w[0].reshape((len(w[0]),1))
 #----------------------------------------------------------------------------------

 #---------- Step 4 - Solve the streamline equation --------------------------------
 # Solve Streamline
 # psi condition
 RHS_psi = sps.lil_matrix.dot(M,w)
 RHS_psi = np.multiply(RHS_psi,bc_2_psi)
 RHS_psi += bc_1_psi
 psi = scipy.sparse.linalg.cg(LHS_psi,RHS_psi,psi, maxiter=1.0e+05, tol=1.0e-05)
 psi = psi[0].reshape((len(psi[0]),1))
 #----------------------------------------------------------------------------------

 #---------- Step 5 - Compute the velocity field -----------------------------------
 #METHOD 1 --> Applying bcs after solve
 # Velocity vx
# AA = sps.lil_matrix.dot(Gy,psi)
# vx = np.dot(Minv,AA)
# vx = np.multiply(vx,bc_2_vx)
# vx += bc_dirichlet_vx

 # Velocity vy
# AA = -sps.lil_matrix.dot(Gx,psi)
# vy = np.dot(Minv,AA)
# vy = np.multiply(vy,bc_2_vy)
# vy += bc_dirichlet_vy

 #METHOD 2 --> Applying bcs before solve
 # Velocity vx
 AA = sps.lil_matrix.dot(Gy,psi)
 RHS_vx = np.multiply(AA,bc_2_vx)
 RHS_vx += bc_1_vx
 vx = scipy.sparse.linalg.cg(LHS_vx,RHS_vx,vx, maxiter=1.0e+05, tol=1.0e-05)
 vx = vx[0].reshape((len(vx[0]),1))
 
 # Velocity vy
 AA = -sps.lil_matrix.dot(Gx,psi)
 RHS_vy = np.multiply(AA,bc_2_vy)
 RHS_vy += bc_1_vy
 vy = scipy.sparse.linalg.cg(LHS_vy,RHS_vy,vy, maxiter=1.0e+05, tol=1.0e-05)
 vy = vy[0].reshape((len(vy[0]),1))
 #----------------------------------------------------------------------------------
