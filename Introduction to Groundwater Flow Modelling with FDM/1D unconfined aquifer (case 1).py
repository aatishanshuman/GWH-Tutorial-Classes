import numpy as np
import matplotlib.pyplot as plt

# Given parameters
L = 5000              # Length of domain (meters)
delx = 1250          # Distance between nodes (meters)
K = 5                 # Hydraulic conductivity (m/d)
Q = 0.0005            # Discharge (m/d)

# Number of nodes
n_nodes = int(L / delx) + 1

# Initialize the head squared vector (h^2)
z = np.zeros(n_nodes)

# Construct the coefficient matrix (tridiagonal system)
A = np.zeros((n_nodes, n_nodes))

# Fill the matrix A with finite difference coefficients
for i in range(1, n_nodes - 1):  # Start from node 1 to n-2 for interior nodes
    A[i, i-1] = 1  # Coefficient for z[i-1]
    A[i, i] = -2   # Coefficient for z[i]
    A[i, i+1] = 1  # Coefficient for z[i+1]

# Boundary conditions for the first and last rows
A[0,0] =1
A[-1,-1] = 1

# Right-hand side vector for the second derivative of z (using -2 * Q / K)
b = np.zeros(n_nodes)

# Set the right-hand side vector for the interior points (using -2 * Q / K)
for i in range(1, n_nodes - 1):
    b[i] = -(2 * Q *delx**2) / K

# Boundary conditions for the first and last points (assuming h^2 boundary conditions)
b[0] = 102**2 
b[-1] = 100 ** 2   # Boundary condition at x=L, h^2 = 100^2 

# Solve the linear system for h^2 (z)
z = np.linalg.solve(A, b)

# To get head h, take the square root of z
h = np.sqrt(z)

# Plot the results
x = np.linspace(0, L, n_nodes)

plt.figure()
plt.rcParams['font.family']='Arial'
plt.plot(x, h)
plt.xlabel('Distance (m)')
plt.ylabel('Head (m)')
plt.title('Head Distribution in the 1D Unconfined Aquifer')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Mark the average head values
plt.scatter(x,[(h[0]+h[-1])*0.5]*len(x))

plt.tight_layout()


ans=A
