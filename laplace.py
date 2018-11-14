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
name_mesh = 'malha_laplace2.msh'
number_equation = 1
mesh = trimsh.Linear('../mesh',name_mesh,number_equation)
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
nt = 10
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



LHS0 = (M/dt) + (theta*K)
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
bc[0][0] = 0.0
bc[1][0] = 0.0
bc[2][0] = 0.0
bc[3][0] = 100.0

# Applying the boundary conditions
bc_neumann,bc_dirichlet,LHS,bc_1,bc_2,ibc = b_bc(mesh.npoints, mesh.x, mesh.y, bc, mesh.neumann_edges[1], mesh.dirichlet_pts[1], LHS0, mesh.neighbors_nodes)


# Initial condition
c = np.zeros([mesh.npoints,1], dtype = float)
'''
for i in range(0,mesh.npoints):
 if mesh.x[i] > 0.1 and\
    mesh.x[i] < 0.3 and\
    mesh.y[i] > 0.4 and\
    mesh.y[i] < 0.6:
 
  c[i] = (np.sin(((mesh.x[i]-0.1)/0.2)*np.pi) + np.sin(((mesh.y[i]-0.4)/0.2)*np.pi))/2.0
'''


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
 save = InOut.Linear(mesh.x,mesh.y,mesh.IEN,mesh.npoints,mesh.nelem,c,vx,vy)
 save.saveVTK('./result_laplace','laplace%s' %t)

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
#print c

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


