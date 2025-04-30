import numpy as np
import matplotlib.pyplot as plt

# Given parameters
L = 5000              # Length of domain (meters)
delx = 1250           # Distance between nodes (meters)
T = 500               # Transmissivity
Q = 0.0005            # Discharge (m/d)


# Number of nodes
n_nodes = int(L / delx) + 1

# Initialize the head vector
h = np.zeros(n_nodes)

# # Boundary conditions
# h[0] = 112.5 # Boundary condition at x=0
# h[-1] = 100  # Boundary condition at x=L 

# Construct the coefficient matrix (tridiagonal system)
A = np.zeros((n_nodes, n_nodes))

# Fill the matrix A with finite difference coefficients
for i in range(1, n_nodes - 1):  # Start from node 1 to n-2 for interior nodes
    A[i, i-1] = 1  # Coefficient for h[i-1]
    A[i, i] = -2   # Coefficient for h[i]
    A[i, i+1] = 1  # Coefficient for h[i+1]

# Boundary conditions for the first and last rows
A[0,0] = -2
A[0,1]=2
A[-1,-1] = 1


# Right-hand side vector
b = np.zeros(n_nodes)

# Set the right-hand side vector for the interior points (using -delx^2 * Q / T)
for i in range(1, n_nodes - 1):
    b[i] = -(delx**2 * Q) / T

# Boundary conditions for the first and last points
b[0] = -(delx**2 * Q) / T
b[-1] = 100  # Boundary condition at x=L 

# Solve the linear system
h = np.linalg.solve(A, b)

# Plot the results
x = np.linspace(0, L, n_nodes)

plt.figure()
plt.rcParams['font.family']='Arial'
plt.plot(x, h)
plt.xlabel('Distance (m)')
plt.ylabel('Head (m)')
plt.title('Head Distribution in the 1D Confined Aquifer')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.scatter(x,[(h[0]+h[-1])*0.5]*len(x))

plt.tight_layout()
