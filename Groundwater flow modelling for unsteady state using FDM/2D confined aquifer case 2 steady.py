import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.animation

#%%

# Given parameters
Lx = 5000             # Length of domain in x-direction (meters)
Ly = 5000             # Length of domain in y-direction (meters)
delx = 500           # Distance between nodes in x-direction (meters)
dely = 500           # Distance between nodes in y-direction (meters)
T = 500              # Transmissivity (m^2/day)
Q = 0.005            # Discharge (m^3/day)

# Number of nodes in each direction
n_nodes_x = int(Lx / delx) + 1
n_nodes_y = int(Ly / dely) + 1

# Initialize node coordinates
nodes = []

# Generate coordinates for nodes in the grid
for i in range(n_nodes_x):
    for j in range(n_nodes_y):
        x_coord = delx * i
        y_coord = dely * j
        nodes.append([x_coord, y_coord])

# Convert the node list to a numpy array
nodes = np.array(nodes)

# Extract x and y coordinates
xcord = nodes[:, 0]
ycord = nodes[:, 1]
numnodes = len(nodes) # Number of nodes

# # Output the number of nodes and the coordinates
# print(f"Number of nodes: {numnodes}")
# print(f"Node coordinates: \n{nodes}")

                          

#%%
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def nodenumber(x,y,nodes, numnodes):
    for i in range(numnodes):
        if nodes[i][0] ==x and nodes[i][1]==y:
            node=i
    return node


left=np.arange(0,n_nodes_x ,1)
right=np.arange(numnodes-n_nodes_y ,numnodes,1)
bottom=np.arange(n_nodes_y,numnodes-n_nodes_y-1, n_nodes_y )
top=np.arange(2*n_nodes_y-1 ,numnodes-1,n_nodes_y )

below_top=np.arange(2*n_nodes_y-2 ,numnodes-2,n_nodes_y )
above_bottom=np.arange(n_nodes_y+1,numnodes-n_nodes_y-1, n_nodes_y )

# top=np.arange(ydiv+ydiv+1,SIZE-1,ydiv+1)
b_nodes=len(left)+len(right)+len(top)+len(bottom)
boundary=np.concatenate((left,right,top,bottom),axis=0)
no_flow_boundary=np.concatenate((top,bottom),axis=0)

#%%

plt.figure()
plt.scatter(nodes[:,0],nodes[:,1])
plt.scatter(nodes[left,0],nodes[left,1],label='Left Boundary')
plt.scatter(nodes[right,0],nodes[right,1], label='Right Boundary')
plt.scatter(nodes[top,0],nodes[top,1], label='Top Boundary')
plt.scatter(nodes[bottom,0],nodes[bottom,1], label='Bottom Boundary')

plt.scatter(nodes[below_top,0],nodes[below_top,1], label='Below_Top Boundary')
plt.scatter(nodes[above_bottom,0],nodes[above_bottom,1], label='Above bottom Boundary')


for i in range(numnodes):
    plt.text(nodes[i,0],nodes[i,1], i)

plt.legend()

#%%

A = np.zeros((numnodes, numnodes))
b = np.zeros(numnodes)

# Interior nodes
for i in range(numnodes):
    x = nodes[i][0]
    y = nodes[i][1]

    # Get node numbers of neighbors (handle boundary cases)

    A[i, i] = -4
    if i in [closest_node([2500,2500], nodes)]:
        b[i] = -delx**2 * Q/T
    else:
        b[i] = 0  # RHS is zero for Laplace equation
    
    if i not in boundary: 
        north_node = nodenumber(x, y + dely, nodes, numnodes)
        south_node = nodenumber(x, y - dely, nodes, numnodes)
        east_node = nodenumber(x + delx, y, nodes, numnodes)
        west_node = nodenumber(x - delx, y, nodes, numnodes)
        
        A[i, i] = -4  # Main diagonal
        if north_node is not None: A[i, north_node] = 1
        if south_node is not None: A[i, south_node] = 1
        if east_node is not None: A[i, east_node] = 1
        if west_node is not None: A[i, west_node] = 1
    if i in top:
        south_node = nodenumber(x, y - dely, nodes, numnodes)
        east_node = nodenumber(x + delx, y, nodes, numnodes)
        west_node = nodenumber(x - delx, y, nodes, numnodes)
        
        if south_node is not None: A[i, south_node] = 2
        if east_node is not None: A[i, east_node] = 1
        if west_node is not None: A[i, west_node] = 1
        
    if i in bottom:
        north_node = nodenumber(x, y + dely, nodes, numnodes)
        east_node = nodenumber(x + delx, y, nodes, numnodes)
        west_node = nodenumber(x - delx, y, nodes, numnodes)
        
        if north_node is not None: A[i, north_node] = 2
        if east_node is not None: A[i, east_node] = 1
        if west_node is not None: A[i, west_node] = 1

        
# Boundary conditions 
 
b[left]=100
A[left,left]=1

b[right]=90
A[right,right]=1

# Solve the system
h = linalg.solve(A, b)

#%%

plt.figure(figsize=(8, 6))
plt.tricontourf(nodes[:,0],nodes[:,1], h, 20, cmap='viridis')
plt.colorbar(label='Solution (u)')
plt.xlabel('x (meters)')
plt.ylabel('y (meters)')
plt.title('Head Distribution')
plt.show()

#%%

# Constants for velocity calculation
k = T  # Transmissivity (k = T)

h_reshaped = h.reshape(n_nodes_y, n_nodes_x)

# Central difference for dh/dx and dh/dy
dh_dx = np.zeros_like(h_reshaped)
dh_dy = np.zeros_like(h_reshaped)

# Apply central difference (excluding boundaries where it would be undefined)
dh_dx[1:-1, :] = (h_reshaped[2:, :] - h_reshaped[:-2, :]) / (2 * delx)
dh_dy[:, 1:-1] = (h_reshaped[:, 2:] - h_reshaped[:, :-2]) / (2 * dely)

# Now compute velocity components (v = k * dh/dx and v = k * dh/dy)
velocity_x = -k * dh_dx
velocity_y = -k * dh_dy

# Flatten the velocities to match the original 1D node array for quiver plot
velocity_x_flat = velocity_x.flatten()
velocity_y_flat = velocity_y.flatten()

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

# Contour plot of h
contour = ax.tricontour(nodes[:, 0], nodes[:, 1], h, 5 ,colors=['k'])

# Add labels to the contours (clabels)
clabels = ax.clabel(contour, inline=True, fontsize=9, fmt='%1.1f')  # Customize fmt as needed

# Quiver plot for velocities
quiver = ax.quiver(nodes[:, 0], nodes[:, 1], velocity_x_flat, velocity_y_flat, scale=50, color='k', width=0.003)

ax.set_xlabel('x (meters)')
ax.set_ylabel('y (meters)')
ax.set_title('Head Distribution with Velocity Vectors')

plt.show()


