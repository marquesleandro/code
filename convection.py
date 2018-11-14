'''
 ==========================================
 Solve the nondimensional equations:

 dc/dt + v.(grad*c) = 0

 for two dimensional space
 using the finite element method
 ==========================================
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

name_mesh = 'malha_convection.msh'
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
nt = 1500 
theta = 1.0

# Nondimensional Numbers
Re = 1000.0
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


#LHS_c0 = (sps.lil_matrix.copy(M)/dt) + (theta/Re)*sps.lil_matrix.copy(K)
LHS_c0 = (sps.lil_matrix.copy(M)/dt)


end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""


print '--------------------------------'
print 'INITIAL AND BOUNDARY CONDITIONS:'
print '--------------------------------'

start_time = time()

# ---------Boundaries conditions--------------------
bc = np.zeros([mesh.nphysical,1], dtype = float) 

# Velocity c condition
bc[0][0] = 0.0
bc[1][0] = 0.0
bc[2][0] = 0.0
bc[3][0] = 0.0


# Applying c condition boundary
bc_neumann_c,bc_dirichlet_c,LHS_c,bc_1_c,bc_2_c,ibc_c = b_bc(mesh.npoints, mesh.x, mesh.y, bc, mesh.neumann_edges[1], mesh.dirichlet_pts[1], LHS_c0, mesh.neighbors_nodes)
#----------------------------------------------------------------------------------

# ---------Initial condition--------------------
c = bc_dirichlet_c
vx = np.zeros([mesh.npoints,1], dtype = float)
vy = np.zeros([mesh.npoints,1], dtype = float)

for i in range(0,mesh.npoints):
 vx[i] = -mesh.y[i] 
 vy[i] = mesh.x[i]

a = 0.0
b = -1.5
r = 1.0

for i in range(0, mesh.npoints):
 x = mesh.x[i] - a
 y = mesh.y[i] - b
 lenght = np.sqrt(x**2 + y**2)

 if lenght < r:
  c[i] = r**2 - (mesh.x[i] - a)**2 - (mesh.y[i] - b)**2



end_time = time()
print 'time duration: %.1f seconds' %(end_time - start_time)
print ""



print '----------------------------'
print 'SOLVE THE LINEARS EQUATIONS:'
print '----------------------------'
print ""

for t in tqdm(range(0, nt)):
 save = InOut.Linear(mesh.x,mesh.y,mesh.IEN,mesh.npoints,mesh.nelem,c,c,c,vx,vy)
 save.saveVTK('./result_convection','convection%s' %t)

 A = np.copy(M)/dt
 RHS_c = sps.lil_matrix.dot(A,c) - np.multiply(vx,sps.lil_matrix.dot(Gx,c))\
       - np.multiply(vy,sps.lil_matrix.dot(Gy,c))\
       - (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(Kxx,c)) + np.multiply(vy,sps.lil_matrix.dot(Kyx,c))))\
       - (dt/2.0)*np.multiply(vy,(np.multiply(vx,sps.lil_matrix.dot(Kxy,c)) + np.multiply(vy,sps.lil_matrix.dot(Kyy,c))))
 RHS_c = np.multiply(RHS_c,bc_2_c)
 RHS_c += bc_1_c
 c = scipy.sparse.linalg.cg(LHS_c,RHS_c,c, maxiter=1.0e+05, tol=1.0e-05)
 c = c[0].reshape((len(c[0]),1))
