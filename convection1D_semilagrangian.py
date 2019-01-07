# ==========================================
# Code created by Leandro Marques at 01/2019
# Gesar Search Group
# State University of the Rio de Janeiro
# e-mail: marquesleandro67@gmail.com
# ==========================================

# This code is used for to use semi-lagrangian scheme in 1D simulation



# =======================
# Importing the libraries
# =======================

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import semi_lagrangian


# ===========
# Import Mesh
# ===========

npoints = 200
nelem = npoints -1
L = 10.0

x = np.zeros([npoints,1], dtype = float)
IEN = np.zeros([nelem,2], dtype = int)

for i in range(0,npoints):
 x[i] = (L/nelem)*i

for e in range(0,nelem):
 IEN[e][0] = e
 IEN[e][1] = e + 1



# ==========
# Parameters
# ==========

Re = 10.0
dt = 0.05
nt = 50

# ========
# Assembly
# ========

K = np.zeros([npoints,npoints], dtype = float)
M = np.zeros([npoints,npoints], dtype = float)
LHS = np.zeros([npoints,npoints], dtype = float)

k_elem = np.array([[1,-1],[-1,1]])
m_elem = np.array([[2,1],[1,2]])

for e in range(0,nelem):
 h = x[e+1]-x[e]

 for i in range(0,2):
  ii = IEN[e][i]

  for j in range(0,2):
   jj = IEN[e][j]

   K[ii][jj] += (1.0/h)*k_elem[i][j]
   M[ii][jj] += (h/6.0)*m_elem[i][j]


#LHS = np.copy(M)/dt + (1.0/Re)*np.copy(K)
LHS = np.copy(M)/dt


# ===================
# Boundary conditions
# ===================

bc_dirichlet = np.zeros([1,npoints], dtype = float)
bc_neumann = np.zeros([npoints,1], dtype = float)
bc_1 = np.zeros([npoints,1], dtype = float)
bc_2 = np.ones([npoints,1], dtype = float)
ibc = []

T = np.zeros([npoints,1],dtype = float)
for i in range(0,npoints):
 if x[i] < 1.0:
  T[i] = abs(np.sin(np.pi*x[i]))

#plt.plot(x, T, '-',color = 'black', label = "numeric solution")
#plt.show()


# Dirichlet condition
#bc_1[0] = 0.0
#bc_1[npoints-1] = 0.0

#ibc.append(0)
#ibc.append(npoints-1)


# Neumann condition
bc_neumann[0] = 0.0
bc_neumann[npoints-1] = 0.0


for mm in ibc:
 bc_dirichlet -= LHS[:,mm]*bc_1[mm]
 LHS[:,mm] = 0.0
 LHS[mm,:] = 0.0
 LHS[mm,mm] = 1.0
 bc_dirichlet[0][mm] = bc_1[mm]
 bc_2[mm] = 0.0

bc_dirichlet = np.transpose(bc_dirichlet)


# =========================
# Solve the linear equation
# =========================

RHS = np.zeros([npoints,1], dtype = float)
v = np.ones([npoints,1], dtype = float)

for t in tqdm(range(0,nt)):
 x_d = x - v*dt

 T_d = semi_lagrangian.Linear_1D(npoints, nelem, IEN, x, x_d, T)

 A = np.copy(M)/dt 
 RHS = np.dot(A,T_d)

 RHS = RHS + (1.0/Re)*bc_neumann
 RHS = np.multiply(RHS,bc_2)
 RHS = RHS + bc_dirichlet

 T = np.linalg.solve(LHS,RHS)

 # Plot Solution
 plt.clf()
 plt.plot(x, T, '-', color = 'black', label = "numeric solution")
 plt.legend(loc = 1)
 plt.xlabel('x')
 plt.ylabel('T')
 plt.pause(1e-100)
