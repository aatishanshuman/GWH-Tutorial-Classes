import numpy as np
import matplotlib.pyplot as plt

# Define problem parameters
L = 1000       # Length of the domain (meters)
delx = 1      # Distance between nodes (spatial step, meters)
D = 100           # diffusion coefficient
V = 1         # Velocity of transport (m/day)
t_total = 1000 # Total simulation time (days)
delt = 1      # Time step (days)
alpha = 0.5   # Implicitness factor (0 = explicit, 1 = fully implicit, 0.5 = Crank-Nicholson)

# Calculate the number of spatial nodes
n_nodes = int(L / delx) + 1  

# Initialize concentration array (initially all zeros)
c_old = np.zeros(n_nodes)

# Initialize a list to store concentration values at all time steps
conc_history = []
conc_history.append(c_old.copy())  # Store initial concentration

# Time loop
t = 0.0  # Initialize time variable
while t < t_total:
    
    A = np.zeros((n_nodes, n_nodes))  # Tridiagonal matrix A
    b = np.zeros(n_nodes)  # Right-hand-side vector b

    # Fill in the interior nodes of matrix A and vector b
    for i in range(1, n_nodes - 1):
    
        A[i, i-1] = -alpha * D * delt / delx**2 -V*delt /(2*delx)
        A[i, i]   = 1 + (2 * alpha * D * delt / delx**2) #+ V*delt /delx
        A[i, i+1] = -alpha * D * delt / delx**2 + V*delt /(2*delx)

     
        b[i] = c_old[i] + (1 - alpha) * ((D * delt / delx**2) * (c_old[i+1] - 2 * c_old[i] + c_old[i-1]) 
            - (V * delt / (2* delx) * (c_old[i+1] - c_old[i-1]) ))


    A[0, 0] = 1  # Dirichlet boundary condition at the left
    b[0] = 100


    A[-1, -1] = 1  # Dirichlet boundary condition at the right
    b[-1] = 0  # Concentration at the right boundary is always zero

    # Solve the linear system Ac = b to get the new concentration values
    c = np.linalg.solve(A, b)

    # Update concentration for the next time step
    c_old = c.copy()

    # Store the concentration values for this time step
    conc_history.append(c.copy())

    # Increment time
    t += delt

# Convert concentration history to a NumPy array for easier plotting
conc_history = np.array(conc_history)

# Create spatial and time arrays for plotting
x = np.linspace(0, L, n_nodes)  # Spatial domain (x-axis)
timesteps = np.arange(0, t_total + delt, delt)  # Time steps (y-axis)

#%%
import matplotlib.cm as cm
plt.figure(figsize=(10, 6))

# Get the spectral colormap
cmap = cm.get_cmap('Spectral')  # or cm.Spectral

# Normalize timesteps for colormap mapping
norm = plt.Normalize(vmin=timesteps.min(), vmax=timesteps.max())

for i, timestep in enumerate(timesteps):
    if i < conc_history.shape[0]:
        # Map the timestep to a color in the spectral colormap
        color = cmap(norm(timestep))
        plt.plot(x, conc_history[i,:], color=color, label=f'Time = {timestep:.0f} days')

plt.xlabel('Distance (m)')
plt.ylabel('conc (m)')
plt.title('Conc Variation Over Time and Space')
plt.grid(True)


#%%


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Define time steps for snapshots (modify as needed)
snapshot_times = [int(t_total/4), int(t_total/2) , int(2*t_total/3) , int(t_total)]  # Choose representative timesteps

# Find indices corresponding to selected timesteps
snapshot_indices = [np.argmin(np.abs(timesteps - t)) for t in snapshot_times]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 grid of subplots
axes = axes.flatten()  # Flatten for easy iteration


# Loop through selected time steps and plot
for i, idx in enumerate(snapshot_indices):
    ax = axes[i]
    ax.plot(x, conc_history[idx], color='k', lw=2)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Concentration")
    ax.set_title(f"Time = {snapshot_times[i]} days")
    ax.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()



#%%



import matplotlib.animation as animation

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(x.min(), x.max())
ax.set_ylim(conc_history.min(), conc_history.max())
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Concentration (m)')
title = ax.set_title(f'Concentration Variation - Day 0')

# Initialize plot with a single color
line, = ax.plot([], [], color='b', lw=2)  # Blue line

# Update function for animation
def update(frame):
    line.set_data(x, conc_history[frame, :])  # Update concentration profile
    title.set_text(f'Concentration Variation - Day {timesteps[frame]:.0f}')  # Update title
    return line, title  # Return updated elements

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(timesteps), interval=2, blit=False)

# Show animation
plt.show()
