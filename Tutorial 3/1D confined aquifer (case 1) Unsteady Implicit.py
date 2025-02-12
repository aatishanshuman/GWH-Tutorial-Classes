import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 5000.0       # Length of domain (meters)
delx = 50.0    # Distance between nodes (meters)
T = 500        # Transmissivity (m^2/day)
S = 0.00001       # Storativity (dimensionless)
Q = 0.00001       # Recharge rate (m/day)
t_total =10   # Total simulation time (days)
delt = 1       # Time step (days)


# Number of nodes
n_nodes = int(L / delx) + 1

# Initialize head and previous time step head

h_old = np.zeros(n_nodes)+100
h_old[0] = 100  # Initialize h_old!
h_old[-1] = 100.0

# Initialize a list to store head values at all time steps
head_history = []
head_history.append(h_old.copy())

# Time loop
t = 0.0
while t < t_total:
    # Build the tridiagonal matrix A and vector b
    A = np.zeros((n_nodes, n_nodes))
    b = np.zeros(n_nodes)

    # Interior nodes
    for i in range(1, n_nodes - 1):
        A[i, i-1] = T / delx**2
        A[i, i] = -2*T / delx**2 + S/delt  # Time term added
        A[i, i+1] = T / delx**2
        b[i] = S/delt * h_old[i] - Q      # RHS with previous time step and source

    # Boundary conditions
    A[0, 0] = 1.0
    b[0] = 100
    A[-1, -1] = 1.0
    b[-1] = 100

    # Solve the system
    h = np.linalg.solve(A, b)

    # Update h_old (Important: Copy, don't assign!)
    h_old = h.copy()

    # Store the head values at the current time step
    head_history.append(h.copy())

    # Increment time
    t += delt

# Convert the head history to a NumPy array for easier plotting
head_history = np.array(head_history)

# Plotting
x = np.linspace(0, L, n_nodes)
timesteps = np.arange(0, t_total + delt, delt)  # Array of time steps (corrected)


#%%
import matplotlib.cm as cm
plt.figure(figsize=(10, 6))

# Get the spectral colormap
cmap = cm.get_cmap('Spectral')  # or cm.Spectral

# Normalize timesteps for colormap mapping
norm = plt.Normalize(vmin=timesteps.min(), vmax=timesteps.max())

for i, timestep in enumerate(timesteps):
    if i < head_history.shape[0]:
        # Map the timestep to a color in the spectral colormap
        color = cmap(norm(timestep))
        plt.plot(x, head_history[i,:], color=color, label=f'Time = {timestep:.0f} days')

plt.xlabel('Distance (m)')
plt.ylabel('Head (m)')
plt.title('Head Variation Over Time and Space')
plt.grid(True)

# Add a colorbar to show the time progression
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # You don't need data for a colorbar
cbar = plt.colorbar(sm, label='Time (days)') # Add label to colorbar

plt.tight_layout()
plt.show()