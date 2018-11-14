'''
 ==========================================
 Solve the unsteady laplace equations:

 Concentration --> dc/dt = div.(grad*c)

 for two dimensional space
 using the finite element method
 ==========================================

 
 ------------------------------------------
 The discrete concentration equation for implicit method will be:
 (M/dt)*(c[n+1] - c[n]) = - K*c[n+1] + bc_neumann
 ((M/dt) + K)*c[n+1] = (M/dt)*c[n] + bc_neumann

 Generalizing:
 ((M/dt) + (theta*K))*c[n+1] = ((M/dt) - (theta - 1)*K)*c[n] + bc_neumann

 where:
  The Backward Euler method (implicit method) --> theta = 1
  The Euler method (explicit method) --> theta = 0
  The Runge-Kutta method --> theta = 0.5


 and, LHS = ((M/dt) + (theta*K)) and RHS = ((M/dt) - (theta - 1)*K)*c + bc_neumann

 Thus, c = solve(LHS,RHS)
 ------------------------------------------
'''


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
name_mesh = 'malha_laplace_professor.msh'
mesh = trimsh.Linear('../mesh',name_mesh)
mesh.parameters(1)
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
nt = 100
theta = 1.0

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
print ""
# -----------------------------------------------



print '--------'
print 'ASSEMBLY:'
print '--------'

start_time = time()
K = sps.lil_matrix((mesh.npoints,mesh.npoints), dtype = float)
M = sps.lil_matrix((mesh.npoints,mesh.npoints), dtype = float)
Gx = sps.lil_matrix((mesh.npoints,mesh.npoints), dtype = float)
Gy = sps.lil_matrix((mesh.npoints,mesh.npoints), dtype = float)

K1 = np.zeros([mesh.npoints,mesh.npoints], dtype = float)
M1 = np.zeros([mesh.npoints,mesh.npoints], dtype = float)
Gx1 = np.zeros([mesh.npoints,mesh.npoints], dtype = float)
Gy1 = np.zeros([mesh.npoints,mesh.npoints], dtype = float)

linear = trielem.Linear(mesh.x, mesh.y, mesh.IEN)

for e in range(0,mesh.nelem): 
# linear.analytic(e)
 linear.numerical(e)

 for i in range(0,3): 
  ii = mesh.IEN[e][i]
  
  for j in range(0,3):
   jj = mesh.IEN[e][j]

   K[ii,jj] += linear.kxx[i][j] + linear.kyy[i][j]
   M[ii,jj] += linear.mass[i][j]

   K1[ii,jj] += linear.kxx[i][j] + linear.kyy[i][j]
   M1[ii,jj] += linear.mass[i][j]

   

  for k in range(0,3):
   kk = mesh.IEN[e][k]

   Gx[ii,kk] += linear.gx[i][k]
   Gy[ii,kk] += linear.gy[i][k]
   
   Gx1[ii,kk] += linear.gx[i][k]
   Gy1[ii,kk] += linear.gy[i][k]



LHS = (M/dt) + (theta*K)
#LHS1 = (M1/dt) + (theta*K1)


end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""




print '--------------------------------'
print 'INITIAL AND BOUNDARY CONDITIONS:'
print '--------------------------------'

start_time = time()

# boundary condition
bc = np.zeros([mesh.nphysical,1], dtype = float) 
bc_dirichlet = np.zeros([mesh.npoints,1], dtype = float) 
bc_neumann = np.zeros([mesh.npoints,1], dtype = float) 
ibc = []

for i in range(0, len(mesh.dirichlet_pts[1])):
 line = mesh.dirichlet_pts[1][i][0] - 1
 v1 = mesh.dirichlet_pts[1][i][1] - 1
 v2 = mesh.dirichlet_pts[1][i][2] - 1

 bc[0] = mesh.x[v1]
 bc[1] = mesh.y[v1]**2 + 1.0
 bc[2] = mesh.x[v1]**2 + 1.0
 bc[3] = mesh.y[v1]

 bc_dirichlet[v1] = bc[line]
 bc_dirichlet[v2] = bc[line]

 ibc.append(v1)
 ibc.append(v2)

bc_dirichlet[0] = 0.0
ibc = np.unique(ibc)


# Gaussian elimination
bc_1 = np.zeros([mesh.npoints,1], dtype = float)
bc_2 = np.ones([mesh.npoints,1], dtype = float)
for i in range(0,len(ibc)):
 mm = ibc[i]
 for nn in mesh.neighbors_nodes[mm]:
  bc_1[nn] -= float(LHS[nn,mm]*bc_dirichlet[mm])
  LHS[nn,mm] = 0.0
  LHS[mm,nn] = 0.0
   
 LHS[mm,mm] = 1.0
 bc_1[mm] = bc_dirichlet[mm]
 bc_2[mm] = 0.0
 

# Initial condition
c = bc_dirichlet

end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""



print '----------------------------'
print 'SOLVE THE LINEARS EQUATIONS:'
print '----------------------------'
print ""

RHS = np.zeros([mesh.npoints,1], dtype = float)
vx = np.zeros([mesh.npoints,1], dtype = float)
vy = np.zeros([mesh.npoints,1], dtype = float)


for t in tqdm(range(0, nt)):
# save = InOut.Linear(mesh.x,mesh.y,mesh.IEN,mesh.npoints,mesh.nelem,c,c,vx,vy)
# save.saveVTK('./result_laplace','laplace%s' %t)

# Scipy Sparse solve
# A = (M/dt) - (theta - 1.0)*K
# RHS = sps.csr_matrix.dot(A,c) + boundary.bc_neumann
# RHS = np.multiply(RHS,boundary.bc_2)
# RHS += boundary.bc_1
# c = scipy.sparse.linalg.spsolve(LHS,RHS)
# c = c.reshape((len(c),1))
#print c

 # Scipy Conjugate Gradient
 A = (M/dt) - (theta - 1.0)*K
 RHS = sps.csr_matrix.dot(A,c) + bc_neumann
 RHS = np.multiply(RHS,bc_2)
 RHS += bc_1
 c = scipy.sparse.linalg.cg(LHS,RHS,c, maxiter=1.0e+02, tol=1.0e-08)
 c = c[0].reshape((len(c[0]),1))
print c

 # Scipy solve for numpy array
# A = (M1/dt) - (theta - 1.0)*K1
# RHS = np.dot(A,c) + boundary.bc_neumann
# RHS = np.multiply(RHS,boundary.bc_2)
# RHS += boundary.bc_1
# c = scipy.linalg.solve(LHS1,RHS)
#print c

 # Numpy solve
# A = (M1/dt) - (theta - 1.0)*K1
# RHS = np.dot(A,c) + boundary.bc_neumann
# RHS = np.multiply(RHS,boundary.bc_2)
# RHS += boundary.bc_1
# c = np.linalg.solve(LHS1,RHS)
#print c

