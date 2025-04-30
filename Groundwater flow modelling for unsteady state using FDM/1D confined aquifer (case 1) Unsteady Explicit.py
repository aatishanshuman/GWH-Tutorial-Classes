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

# Number of nodes (same as before)
n_nodes = int(L / delx) + 1

# Initialize head and previous time step head (same as before)
h = np.zeros(n_nodes) + 100.0
h[0] = 100.0
h[-1] = 100.0
h_old = h.copy()  # Important: Copy, not assign!

# Initialize a list to store head values at all time steps
head_history = []
head_history.append(h.copy())

# Time loop
t = 0.0
while t < t_total:
    # Interior nodes (Explicit update)
    for i in range(1, n_nodes - 1):
        h[i] = h_old[i] + delt * (T / delx**2 * (h_old[i-1] - 2*h_old[i] + h_old[i+1]) + Q/S)

    # Boundary conditions (same as before)
    h[0] = 100.0
    h[-1] = 100.0

    # Store the head values at the current time step
    head_history.append(h.copy())

    # Update h_old (Important: Copy, don't assign!)
    h_old = h.copy()

    # Increment time
    t += delt

# Convert the head history to a NumPy array for easier plotting
head_history = np.array(head_history)

# Plotting (same as before)
x = np.linspace(0, L, n_nodes)
timesteps = np.arange(0, t_total + delt, delt)

#%%
import matplotlib.cm as cm
plt.figure(figsize=(10, 6))

cmap = cm.get_cmap('Spectral')
norm = plt.Normalize(vmin=timesteps.min(), vmax=timesteps.max())

for i, timestep in enumerate(timesteps):
    if i < head_history.shape[0]:
        color = cmap(norm(timestep))
        plt.plot(x, head_history[i,:], color=color, label=f'Time = {timestep:.0f} days')

plt.xlabel('Distance (m)')
plt.ylabel('Head (m)')
plt.title('Head Variation Over Time and Space (Explicit)')
plt.grid(True)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, label='Time (days)')

plt.tight_layout()
plt.show()
