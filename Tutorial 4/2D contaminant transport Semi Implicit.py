import numpy as np
from scipy import linalg

# Parameters
Lx, Ly = 200, 200
delx, dely = 10, 10
t_total, delt = 100, 1  # Adjust for stability!
alpha, rate = 0.5, 0.005
Dxx, Dyy = 10, 10
Vx, Vy, R = 1, 1, 1
porosity=0.3
depth=30

# Number of nodes
n_nodes_x, n_nodes_y = int(Lx / delx) + 1, int(Ly / dely) + 1
numnodes = n_nodes_x * n_nodes_y

# Node coordinates
nodes = np.array([[delx * i, dely * j] for i in range(n_nodes_x) for j in range(n_nodes_y)])

# Boundary node indices
left = np.arange(0, n_nodes_x, 1)
right = np.arange(numnodes - n_nodes_y, numnodes, 1)
bottom = np.arange(n_nodes_y, numnodes - n_nodes_y - 1, n_nodes_y)
top = np.arange(2 * n_nodes_y - 1, numnodes - 1, n_nodes_y)
boundary = np.concatenate((left, right, top, bottom), axis=0)

# Find the closest node
def closest_node(node, nodes):
    deltas = nodes - node
    return np.argmin(np.einsum('ij,ij->i', deltas, deltas))

# Transport function
def transport(well_locs, rates , concs):
    c_old = np.zeros(numnodes)
    conc_history, time_history = [], []
    
    import sys
    t = 0.0
    while t <= t_total:
        sys.stdout.write(f"\r⏳ Simulating... Day {t:.0f}/{t_total} ({(t/t_total)*100:.1f}%)")
        sys.stdout.flush()

      
        
        A = np.zeros((numnodes, numnodes))
        b = np.zeros(numnodes)

        for i in range(numnodes):
            x, y = nodes[i]

            A[i, i] = 1 + (2 * alpha * (Dxx/R) * delt / delx**2) + (2 * alpha * (Dyy/R) * delt / dely**2) + (rate/R)

            if i not in boundary:
                north, south = closest_node([x, y + dely], nodes), closest_node([x, y - dely], nodes)
                east, west = closest_node([x + delx, y], nodes), closest_node([x - delx, y], nodes)

                A[i, north] = -alpha * (Dyy/R) * delt / delx**2 + (Vy/R) * delt / (2*dely)
                A[i, south] = -alpha * (Dyy/R) * delt / delx**2 - (Vy/R) * delt / (2*dely)
                A[i, east] = -alpha * (Dxx/R) * delt / delx**2 + (Vx/R) * delt / (2*delx)
                A[i, west] = -alpha * (Dxx/R) * delt / delx**2 - (Vx/R) * delt / (2*delx)

                diffusion = (Dxx/R) * delt / delx**2 * (c_old[east] - 2*c_old[i] + c_old[west]) + \
                            (Dyy/R) * delt / dely**2 * (c_old[north] - 2*c_old[i] + c_old[south])
                advection = - (Vx/R) * delt / (2*delx) * (c_old[east] - c_old[west]) - \
                            (Vy/R) * delt / (2*dely) * (c_old[north] - c_old[south])

                b[i] = c_old[i] + (1 - alpha) * (diffusion + advection - (rate/R) * c_old[i])

        # Injection well
        for w in range(len(well_locs)):
            b[closest_node(well_locs[w], nodes)] +=  concs[w]* rates[w] /(porosity*depth)
            
    
        

        # Boundary conditions
        for boundary_nodes in [left, right, top, bottom]:
            b[boundary_nodes] = 0
            A[boundary_nodes, boundary_nodes] = 1

        # Solve the system
        h = linalg.solve(A, b)
        conc_history.append(h.copy())
        time_history.append(t)

        c_old = h.copy()
        t += delt
        
        

    return np.array(conc_history), np.array(time_history)

well_locs=[[20,50],[50,10]]
rates=[0.005, 0.01]
concs=[1000,2000]


# Run the transport function
conc_history, time_history = transport(well_locs, rates , concs)



#%%
import matplotlib.pyplot as plt
# Plot the last time step
plt.figure(figsize=(8, 6))
plt.tricontourf(nodes[:, 0], nodes[:, 1], conc_history[-1], cmap='viridis')
plt.colorbar(label='Head (m)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title(f'Concentration Distribution at t = {time_history[-1]:.0f} days')
plt.show()


#%%
# Example: Plotting concentation at a specific node over time:

plt.figure(figsize=(8, 6))
plt.plot(time_history, conc_history[:, closest_node([50.50], nodes)])
plt.xlabel('Time (days)')
plt.ylabel('Concentration (m)')
plt.title('Concentration at Node {} over Time'.format(closest_node([50.50], nodes)))
plt.grid(True)
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation

# Create a triangulation
triang = tri.Triangulation(nodes[:, 0], nodes[:, 1])

# Initialize the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.tricontourf(triang, conc_history[0], cmap='viridis')
plt.tight_layout()

# Add colorbar
cbar = plt.colorbar(contour, ax=ax, label='Concentration')

# Labels and title
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
title = ax.set_title(f'Concentration Distribution at t = {time_history[0]:.0f} days')

# Function to update the animation
def update(frame):
    ax.clear()
    contour = ax.tricontourf(triang, conc_history[frame], cmap='viridis')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Concentration Distribution at t = {time_history[frame]:.0f} days')
    plt.tight_layout()
    return contour

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(time_history), interval=50, repeat=True)

# Save or Show the animation
plt.show()

