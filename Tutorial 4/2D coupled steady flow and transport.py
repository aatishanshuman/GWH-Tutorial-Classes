import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings("ignore")

def closest_node(node, nodes):
    """Find the index of the closest node to a given point."""
    deltas = nodes - node
    return np.argmin(np.einsum('ij,ij->i', deltas, deltas))

# Parameters
Lx, Ly = 200, 200  # Domain size
delx, dely = 5, 5  # Grid spacing
T = 10 # Transmissivity
t_total, delt, alpha = 365*5,5 , 0.5  # Time parameters
alphaL,alphaT=10,1
R=1         #Retardation factor
rate=0  #decay rate
porosity=0.3
depth=30



# Number of nodes
n_nodes_x, n_nodes_y = int(Lx / delx) + 1, int(Ly / dely) + 1
numnodes = n_nodes_x * n_nodes_y

# Generate grid nodes
nodes = np.array([[i * delx, j * dely] for i in range(n_nodes_x) for j in range(n_nodes_y)])

# Define boundary node indices
left, right = np.arange(0, n_nodes_x, 1), np.arange(numnodes - n_nodes_y, numnodes, 1)
bottom, top = np.arange(n_nodes_y, numnodes - n_nodes_y - 1, n_nodes_y), np.arange(2 * n_nodes_y - 1, numnodes - 1, n_nodes_y)
boundary = np.concatenate((left, right, top, bottom), axis=0)

def gwflow(well_locs, rates):
        
    A = np.zeros((numnodes, numnodes))
    b = np.zeros(numnodes)

    # Interior nodes
    for i in range(numnodes):
        x = nodes[i][0]
        y = nodes[i][1]

        # Get node numbers of neighbors (handle boundary cases)

        A[i, i] = -4
          
        
        north_node = closest_node([x, y + dely], nodes)
        south_node = closest_node([x, y - dely], nodes)
        east_node = closest_node([x + delx, y], nodes)
        west_node = closest_node([x - delx, y], nodes)
        
        
        if i not in boundary: 

            A[i, i] = -4  # Main diagonal
            if north_node is not None: A[i, north_node] = 1
            if south_node is not None: A[i, south_node] = 1
            if east_node is not None: A[i, east_node] = 1
            if west_node is not None: A[i, west_node] = 1
        if i in top:
        
            if south_node is not None: A[i, south_node] = 2
            if east_node is not None: A[i, east_node] = 1
            if west_node is not None: A[i, west_node] = 1
            
        if i in bottom:

            
            if north_node is not None: A[i, north_node] = 2
            if east_node is not None: A[i, east_node] = 1
            if west_node is not None: A[i, west_node] = 1
            

        
    for (wx, wy), rate in zip(well_locs, rates):
        b[closest_node([wx, wy], nodes)] +=  -delx**2 * -rate/T

            
    # Boundary conditions 
     
    b[left]=100
    A[left,left]=1

    b[right]=80
    A[right,right]=1

    # Solve the system
    h = linalg.solve(A, b)
    # Compute Darcy velocities
    Vx, Vy = np.zeros(numnodes), np.zeros(numnodes)
    for i in range(numnodes):
        x, y = nodes[i]
        if i not in boundary:
            east, west = closest_node([x + delx, y], nodes), closest_node([x - delx, y], nodes)
            north, south = closest_node([x, y + dely], nodes), closest_node([x, y - dely], nodes)
   
            Vx[i] = -T /depth * (h[east] - h[west]) / (2 * delx)
            Vy[i] = -T /depth * (h[north] - h[south]) / (2 * dely)
   
    return h, Vx, Vy

# Define wells and their rates
well_locs=[[20,100],[80,60] , [150,140]]
rates=[-10, 20 ,20]

# Run groundwater flow simulation
head, Vx, Vy= gwflow(well_locs, rates)


#%%

plt.figure(figsize=(8, 6))
plt.tricontourf(nodes[:,0],nodes[:,1], head, 20, cmap='viridis')
plt.colorbar(label='Solution (u)')
plt.xlabel('x (meters)')
plt.ylabel('y (meters)')
plt.title('Head Distribution')
plt.show()






#%%



# Transport function
def transport(Vx, Vy, well_locs, rates , concs):
    c_old = np.zeros(numnodes)
    conc_history, time_history = [], []

    t = 0.0

    while t <= t_total:
        sys.stdout.write(f"\r⏳ Simulating contaminant transport... Day {t:.0f}/{t_total} ({(t/t_total)*100:.1f}%)")
        sys.stdout.flush()
        

        
        Dxx= (alphaL* Vx**2 + alphaT*Vy**2) /(np.sqrt(Vx**2 + Vy**2))
        Dyy= (alphaL* Vy**2 + alphaT*Vx**2) /(np.sqrt(Vx**2 + Vy**2))
        
     
    
        A = np.zeros((numnodes, numnodes))
        b = np.zeros(numnodes)

        for i in range(numnodes):
            x, y = nodes[i]

            A[i, i] = 1 + (2 * alpha * (Dxx[i]/R) * delt / delx**2) + (2 * alpha * (Dyy[i]/R) * delt / dely**2) + (rate/R)

            if i not in boundary:
                north, south = closest_node([x, y + dely], nodes), closest_node([x, y - dely], nodes)
                east, west = closest_node([x + delx, y], nodes), closest_node([x - delx, y], nodes)

                A[i, north] = -alpha * (Dyy [i]/R) * delt / delx**2 + (Vy[i]/R) * delt / (2*dely)
                A[i, south] = -alpha * (Dyy [i]/R) * delt / delx**2 - (Vy[i]/R) * delt / (2*dely)
                A[i, east] = -alpha * (Dxx [i]/R) * delt / delx**2 + (Vx[i]/R) * delt / (2*delx)
                A[i, west] = -alpha * (Dxx [i]/R) * delt / delx**2 - (Vx[i]/R) * delt / (2*delx)

                diffusion = (Dxx[i]/R) * delt / delx**2 * (c_old[east] - 2*c_old[i] + c_old[west]) + \
                            (Dyy[i]/R) * delt / dely**2 * (c_old[north] - 2*c_old[i] + c_old[south])
                advection = - (Vx[i]/R) * delt / (2*delx) * (c_old[east] - c_old[west]) - \
                            (Vy[i]/R) * delt / (2*dely) * (c_old[north] - c_old[south])

                b[i] = c_old[i] + (1 - alpha) * (diffusion + advection - (rate/R) * c_old[i])

        # Injection well
        for w in range(len(well_locs)):
            b[closest_node(well_locs[w], nodes)] +=  concs[w]* rates[w] /(porosity*depth)
            
    
        

        # Boundary conditions
        for boundary_nodes in [left, right, top, bottom]:
            b[boundary_nodes] = 0
            A[boundary_nodes, boundary_nodes] = 1

        # Solve the system
        ans = linalg.solve(A, b)
        conc_history.append(ans.copy())
        time_history.append(t)

        c_old = ans.copy()
        t += delt
        
        

    return np.array(conc_history), np.array(time_history)


concs=[1000,2000]
well_locs=[[20,100]]
rates=[0.05]



# Run the transport function
conc_history, time_history = transport(Vx, Vy, well_locs, rates , concs)



#%%
import matplotlib.pyplot as plt
# Plot the last time step
plt.figure(figsize=(8, 6))
plt.tricontourf(nodes[:, 0], nodes[:, 1], conc_history[2], cmap='viridis')
plt.colorbar(label='Head (m)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title(f'Concentation Distribution at t = {time_history[-1]:.0f} days')
plt.show()


#%%
# Example: Plotting concentation at a specific node over time:

plt.figure(figsize=(8, 6))
plt.plot(time_history, conc_history[:, closest_node([50,50], nodes)])
plt.xlabel('Time (days)')
plt.ylabel('Ceoncentration (ppm)')
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



