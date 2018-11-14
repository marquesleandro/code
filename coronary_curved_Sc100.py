'''
 ==========================================
 Solve the nondimensional equations:

 Vorticity --> dw/dt + v.(grad*w) = (1.0/Re)*div.(grad*w)
 Streamline --> div.(grad*psi) = -w
 Velocity --> vx = dpsi/dy and vy = -dpsi/dx

 for two dimensional space
 using the finite element method
 ==========================================

 
 ------------ discretization of the vorticity equation ------------------------------
 The discrete concentration equation for implicit method will be:
 (M/dt)*(w[n+1] - w[n]) + v.G*w[n] = (1.0/Re)*K*w[n+1] + bc_neumann
 ((M/dt) + (1.0/Re)*K)*w[n+1] = (M/dt)*w[n] - v.G*w[n] + (1.0/(Re))*bc_neumann

 Generalizing:
 ((M/dt) + (theta/Re)*K)*w[n+1] = ((M/dt) - (theta - 1)*K)*w[n] - v.G*w[n] + (1.0/(Re))*bc_neumann

 where:
  The Backward Euler method (implicit method) --> theta = 1
  The Euler method (explicit method) --> theta = 0
  The Runge-Kutta method --> theta = 0.5


 and, LHS_w = ((M/dt) + (theta/Re)*K) and RHS_w = ((M/dt) - (theta - 1)*K)*w - v.G*w + (1.0/Re)*bc_neumann

 Thus, w = solve(LHS_w,RHS_w)
 ------------------------------------------------------------------------------------


 ------------ discretization of the streamline equation ------------------------------
 K*psi = M*w + bc_neumann

 and, LHS_psi = K and RHS_psi = M*w + bc_neumann

 Thus, psi = solve(LHS_psi,RHS_psi)
 ------------------------------------------------------------------------------------


 ------------ discretization of the velocity equation ------------------------------
 M*vx = Gy*psi
 M*vy = -Gx*psi

 where for vx, LHS_vx = M and RHS_vx = Gy*psi
 where for vy, LHS_vy = M and RHS_vy = -Gx*psi

 Thus for vx,  vx = solve(LHS_vx,RHS_vx)
 Thus for vy,  vy = solve(LHS_vy,RHS_vy)
 ------------------------------------------------------------------------------------
'''


# =======================
# Importing the libraries
# =======================

import sys
sys.path.insert(0, '../lib_class')

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

name_mesh = 'RealGeoStrut_ext.msh'
mesh = trimsh.Linear('../mesh/coronaria',name_mesh)
mesh.parameters(4)
mesh.ien()
mesh.coord()

end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""



# ==========
# Parameters
# ==========

# Time
dt = 0.0001
nt = 100000
theta = 1.0

# Nondimensional Numbers
Re = 54.5
Sc = 100.0
	
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

for e in tqdm(range(0, mesh.nelem)):
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


LHS_vx0 = sps.lil_matrix.copy(M)
LHS_vy0 = sps.lil_matrix.copy(M)
LHS_psi0 = sps.lil_matrix.copy(K)
LHS_c0 = ((sps.lil_matrix.copy(M)/dt) + (theta/(Re*Sc))*sps.lil_matrix.copy(K))


end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""


print '--------------------------------'
print 'INITIAL AND BOUNDARY CONDITIONS:'
print '--------------------------------'

start_time = time()

# ---------Boundaries conditions--------------------
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

'''
# Streamline condition
bc[8][0] = 0.0
bc[9][0] = 0.0
bc[10][0] = 0.0
bc[11][0] = 1.0
'''


# Concentration condition
bc[12][0] = 0.0
bc[13][0] = 0.0
bc[14][0] = 0.0
bc[15][0] = 0.0
bc[16][0] = 1.0

# Applying vx condition
bc_neumann_vx,bc_dirichlet_vx,LHS_vx,bc_1_vx,bc_2_vx,ibc_vx = b_bc(mesh.npoints, mesh.x, mesh.y, bc, mesh.neumann_edges[1], mesh.dirichlet_pts[1], LHS_vx0, mesh.neighbors_nodes)
ibc_w = ibc_vx

# Applying vy condition
bc_neumann_vy,bc_dirichlet_vy,LHS_vy,bc_1_vy,bc_2_vy,ibc_vy = b_bc(mesh.npoints, mesh.x, mesh.y, bc, mesh.neumann_edges[2], mesh.dirichlet_pts[2], LHS_vy0, mesh.neighbors_nodes)

'''
# Applying psi condition
bc_neumann_psi,bc_dirichlet_psi,LHS_psi,bc_1_psi,bc_2_psi,ibc_psi = b_bc(mesh.npoints, mesh.x, mesh.y, bc, mesh.neumann_edges[3], mesh.dirichlet_pts[3], LHS_psi0, mesh.neighbors_nodes)
'''

# Applying concentration condition
bc_neumann_c,bc_dirichlet_c,LHS_c,bc_1_c,bc_2_c,ibc_c = b_bc(mesh.npoints, mesh.x, mesh.y, bc, mesh.neumann_edges[4], mesh.dirichlet_pts[4], LHS_c0, mesh.neighbors_nodes)
#----------------------------------------------------------------------------------

'''
# Applying vx condition (dirichlet condition only)
bc_dirichlet_vx = np.zeros([mesh.npoints,1], dtype = float) 
bc_neumann_vx = np.zeros([mesh.npoints,1], dtype = float) 
ibc_vx = []

v_max = 1.5
L = 1.0
for i in range(0, len(mesh.dirichlet_pts[1])):
 line = mesh.dirichlet_pts[1][i][0] - 1
 v1 = mesh.dirichlet_pts[1][i][1] - 1
 v2 = mesh.dirichlet_pts[1][i][2] - 1

 if line == 0: 
  bc_dirichlet_vx[v1] = 0.0
  bc_dirichlet_vx[v2] = 0.0

  ibc_vx.append(v1)
  ibc_vx.append(v2)

 elif line == 2:
  bc_dirichlet_vx[v1] = v_max*(1.0 - (mesh.y[v1]/L)**2)
  bc_dirichlet_vx[v2] = v_max*(1.0 - (mesh.y[v2]/L)**2)


  ibc_vx.append(v1)
  ibc_vx.append(v2)

ibc_vx = np.unique(ibc_vx)
ibc_w = ibc_vx

# Gaussian elimination for vx
bc_1_vx = np.zeros([mesh.npoints,1], dtype = float)
bc_2_vx = np.ones([mesh.npoints,1], dtype = float)
LHS_vx = sps.lil_matrix.copy(LHS_vx0)
for i in range(0,len(ibc_vx)):
 mm = ibc_vx[i]
 for nn in mesh.neighbors_nodes[mm]:
  bc_1_vx[nn] -= float(LHS_vx[nn,mm]*bc_dirichlet_vx[mm])
  LHS_vx[nn,mm] = 0.0
  LHS_vx[mm,nn] = 0.0
   
 LHS_vx[mm,mm] = 1.0
 bc_1_vx[mm] = bc_dirichlet_vx[mm]
 bc_2_vx[mm] = 0.0
'''

# Applying psi condition (dirichlet condition only)
bc_dirichlet_psi = np.zeros([mesh.npoints,1], dtype = float) 
bc_neumann_psi = np.zeros([mesh.npoints,1], dtype = float) 
ibc_psi = []

for i in range(0, len(mesh.dirichlet_pts[3])):
 line = mesh.dirichlet_pts[3][i][0] - 1
 v1 = mesh.dirichlet_pts[3][i][1] - 1
 v2 = mesh.dirichlet_pts[3][i][2] - 1

 if line == 8:  
  bc_dirichlet_psi[v1] = 0.0
  bc_dirichlet_psi[v2] = 0.0

  ibc_psi.append(v1)
  ibc_psi.append(v2)

 elif line == 11:  
  bc_dirichlet_psi[v1] = 1.0
  bc_dirichlet_psi[v2] = 1.0

  ibc_psi.append(v1)
  ibc_psi.append(v2)

 else:
  bc_dirichlet_psi[v1] = mesh.y[v1]
  bc_dirichlet_psi[v2] = mesh.y[v2]

  ibc_psi.append(v1)
  ibc_psi.append(v2)

ibc_psi = np.unique(ibc_psi)

# Gaussian elimination for psi
bc_1_psi = np.zeros([mesh.npoints,1], dtype = float)
bc_2_psi = np.ones([mesh.npoints,1], dtype = float)
LHS_psi = sps.lil_matrix.copy(LHS_psi0)
for i in range(0,len(ibc_psi)):
 mm = ibc_psi[i]
 for nn in mesh.neighbors_nodes[mm]:
  bc_1_psi[nn] -= float(LHS_psi[nn,mm]*bc_dirichlet_psi[mm])
  LHS_psi[nn,mm] = 0.0
  LHS_psi[mm,nn] = 0.0
   
 LHS_psi[mm,mm] = 1.0
 bc_1_psi[mm] = bc_dirichlet_psi[mm]
 bc_2_psi[mm] = 0.0


# ---------Initial condition--------------------
vx = bc_dirichlet_vx
vy = bc_dirichlet_vy
psi = bc_dirichlet_psi
c = bc_dirichlet_c
w = np.zeros([mesh.npoints,1], dtype = float)


#---------- Step 1 - Compute the vorticity and stream field --------------------
# -----Vorticity initial-----
AA = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
w = scipy.sparse.linalg.cg(M,AA,w, maxiter=1.0e+05, tol=1.0e-05)
w = w[0].reshape((len(w[0]),1))


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


bc_dirichlet_w = np.zeros([mesh.npoints,1], dtype = float) 
for t in tqdm(range(0, nt)):
 save = InOut.Linear(mesh.x,mesh.y,mesh.IEN,mesh.npoints,mesh.nelem,c,w,psi,vx,vy)
 save.saveVTK('./result_coronary_RealGeoStrut_Sc100','coronary%s' %t)

 #---------- Step 2 - Compute the boundary conditions for vorticity --------------
 AA = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
 bc_dirichlet_w = scipy.sparse.linalg.cg(M,AA,bc_dirichlet_w, maxiter=1.0e+05, tol=1.0e-05)
 bc_dirichlet_w = bc_dirichlet_w[0].reshape((len(bc_dirichlet_w[0]),1))


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
       - (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(Kxx,w)) + np.multiply(vy,sps.lil_matrix.dot(Kyx,w))))\
       - (dt/2.0)*np.multiply(vy,(np.multiply(vx,sps.lil_matrix.dot(Kxy,w)) + np.multiply(vy,sps.lil_matrix.dot(Kyy,w))))
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

 #---------- Step 6 - Concentration -----------------------------------
 A = np.copy(M)/dt
 RHS_c = sps.lil_matrix.dot(A,c) - np.multiply(vx,sps.lil_matrix.dot(Gx,c))\
       - np.multiply(vy,sps.lil_matrix.dot(Gy,c)) + (1.0/(Re*Sc))*bc_neumann_c\
       - (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(Kxx,c)) + np.multiply(vy,sps.lil_matrix.dot(Kyx,c))))\
       - (dt/2.0)*np.multiply(vy,(np.multiply(vx,sps.lil_matrix.dot(Kxy,c)) + np.multiply(vy,sps.lil_matrix.dot(Kyy,c))))
 RHS_c = np.multiply(RHS_c,bc_2_c)
 RHS_c += bc_1_c
 c = scipy.sparse.linalg.cg(LHS_c,RHS_c,c, maxiter=1.0e+05, tol=1.0e-05)
 c = c[0].reshape((len(c[0]),1))
 #----------------------------------------------------------------------------------
