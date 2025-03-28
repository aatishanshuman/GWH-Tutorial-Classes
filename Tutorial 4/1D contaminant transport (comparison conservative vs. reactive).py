import numpy as np
import matplotlib.pyplot as plt

def contaminant_transport(L, delx, D, V, t_total, delt, alpha, rate, R):
    """
    Simulates contaminant transport using the finite difference method.
    
    Returns:
    - x: Spatial domain array
    - timesteps: Time step array
    - conc_history: Concentration values over time and space
    """
    n_nodes = int(L / delx) + 1
    x = np.linspace(0, L, n_nodes)  # Spatial domain
    timesteps = np.arange(0, t_total + delt, delt)  # Time steps
    c_old = np.zeros(n_nodes)  # Initial concentration
    conc_history = [c_old.copy()]  # Store concentration history

    # Time loop
    t = 0.0
    while t < t_total:
        A = np.zeros((n_nodes, n_nodes))  # Tridiagonal matrix A
        b = np.zeros(n_nodes)  # Right-hand-side vector b

        for i in range(1, n_nodes - 1):
            A[i, i-1] = -alpha * (D/R) * delt / delx**2 - (V/R) * delt / (2*delx)
            A[i, i]   = 1 + (2 * alpha * (D/R) * delt / delx**2) + (rate/R)
            A[i, i+1] = -alpha * (D/R) * delt / delx**2 + (V/R) * delt / (2*delx)

            b[i] = (c_old[i] +
                   (1 - alpha) * ((D/R) * delt / delx**2 * (c_old[i+1] - 2*c_old[i] + c_old[i-1])
                   - (V/R) * delt / (2*delx) * (c_old[i+1] - c_old[i-1]) 
                   - (rate/R) * c_old[i]))

        # Boundary conditions
        A[0, 0] = 1  # Dirichlet boundary condition (left)
        b[0] = 100  # Fixed concentration at the left boundary

        A[-1, -1] = 1  # Dirichlet boundary condition (right)
        b[-1] = 0  # Fixed concentration at the right boundary

        # Solve the system Ac = b
        c = np.linalg.solve(A, b)

        # Store results
        c_old = c.copy()
        conc_history.append(c.copy())
        t += delt

    return x, timesteps, np.array(conc_history)

# Problem parameters
L = 1000       # Length of the domain (m)
delx = 1       # Spatial step (m)
t_total = 1000 # Total simulation time (days)
delt = 1       # Time step (days)
alpha = 0.5    # Implicitness factor

# Different parameter sets
parameter_sets = [
    {"D": 1, "V": 1, "rate": 0, "R": 1},  # Case 1
    {"D": 10, "V": 1, "rate": 0, "R": 1},  # Case 2 (higher diffusion)
    {"D": 10, "V": 2, "rate": 0., "R":1 },  # Case 3 (higher velocity)
    {"D": 1, "V": 1, "rate": 0.001, "R": 1},  # Case 4 (higher reaction rate)
    {"D": 1, "V": 1, "rate": 0.001, "R": 3}  # Case 5 (higher retardation factor)
]

# Snapshot times
snapshot_times = [int(t_total / 4), int(t_total / 2), int(2 * t_total / 3), int(t_total)]

# Dictionary to store results for each parameter set
results = {}

# Run model once per parameter set and store results
for params in parameter_sets:
    print(f"Running simulation for: {params}")
    x, timesteps, conc_history = contaminant_transport(L, delx, params["D"], params["V"], t_total, delt, alpha, params["rate"], params["R"])
    results[tuple(params.values())] = (x, timesteps, conc_history)  # Store using tuple key

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid
axes = axes.flatten()  # Convert to 1D array for easy iteration
colors = ['b', 'g', 'r', 'm', 'c']  # Different colors for different parameter sets

# Plot results
for idx, snapshot in enumerate(snapshot_times):
    ax = axes[idx]  # Select subplot
    ax.set_title(f'Concentration Profile at Day {snapshot}')

    for p_idx, params in enumerate(parameter_sets):
        param_tuple = tuple(params.values())  # Convert params to tuple for lookup
        x, timesteps, conc_history = results[param_tuple]  # Retrieve precomputed results

        # Get index of the snapshot time
        time_idx = np.where(timesteps == snapshot)[0][0]  

        # Plot
        ax.plot(x, conc_history[time_idx, :], color=colors[p_idx], linestyle='-', alpha=0.8, 
                label=f"V={params['V']}, D={params['D']}, rate={params['rate']}, R={params['R']}")

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Concentration')
    ax.grid(True)
    ax.legend(fontsize=7, loc='upper right')

# Adjust layout
plt.tight_layout()
plt.show()
