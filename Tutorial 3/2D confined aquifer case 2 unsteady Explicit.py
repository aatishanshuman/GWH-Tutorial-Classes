import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.animation


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


# Parameters
Lx = 2000
Ly = 2000
delx = 200
dely = 200
T = 500
S = 0.00001
Q = 0.001
t_total = 10
delt = 0.1  # Adjust for stability!

# Number of nodes
n_nodes_x = int(Lx / delx) + 1
n_nodes_y = int(Ly / dely) + 1
numnodes = n_nodes_x * n_nodes_y

# Node coordinates
nodes = []
for i in range(n_nodes_x):
    for j in range(n_nodes_y):
        x_coord = delx * i
        y_coord = dely * j
        nodes.append([x_coord, y_coord])
nodes = np.array(nodes)

# Boundary node indices
left = np.arange(0, n_nodes_x, 1)
right = np.arange(numnodes - n_nodes_y, numnodes, 1)
bottom = np.arange(n_nodes_y, numnodes - n_nodes_y - 1, n_nodes_y)
top = np.arange(2 * n_nodes_y - 1, numnodes - 1, n_nodes_y)
boundary = np.concatenate((left, right, top, bottom), axis=0)

# Initialize head
h = np.zeros(numnodes)
h_old = np.zeros(numnodes)
h_old[left] = 100
h_old[right] = 90
h_old[top] = 0
h_old[bottom] = 0

# Store head values at all time steps
head_history = []
time_history = []

# Time loop
t = 0.0
while t <= t_total:
    print(t)
    A = np.zeros((numnodes, numnodes))
    b = np.zeros(numnodes)

    # Interior nodes
    for i in range(numnodes):
        x = nodes[i][0]
        y = nodes[i][1]

        A[i, i] = -4 * T / delx**2 - S / delt
        if i in [closest_node([2500, 2500], nodes)]:
            b[i] = -delx**2 * Q / T + S / delt * h_old[i]
        else:
            b[i] = S / delt * h_old[i]

        if i not in boundary:
            # Use closest_node for finding neighbors
            north_node = closest_node([x, y + dely], nodes)
            south_node = closest_node([x, y - dely], nodes)
            east_node = closest_node([x + delx, y], nodes)
            west_node = closest_node([x - delx, y], nodes)

            A[i, i] = -4 * T / delx**2 - S / delt  # Self coefficient for interior nodes
            A[i, north_node] = T / delx**2
            A[i, south_node] = T / delx**2
            A[i, east_node] = T / delx**2
            A[i, west_node] = T / delx**2

        if i in top:
            south_node = closest_node([x, y - dely], nodes)
            east_node = closest_node([x + delx, y], nodes)
            west_node = closest_node([x - delx, y], nodes)

            A[i, south_node] = 2 * T / delx**2
            A[i, east_node] = T / delx**2
            A[i, west_node] = T / delx**2

        if i in bottom:
            north_node = closest_node([x, y + dely], nodes)
            east_node = closest_node([x + delx, y], nodes)
            west_node = closest_node([x - delx, y], nodes)

            A[i, north_node] = 2 * T / delx**2
            A[i, east_node] = T / delx**2
            A[i, west_node] = T / delx**2

    # Boundary conditions
    b[left] = 100
    A[left, left] = 1

    b[right] = 90
    A[right, right] = 1

    # Solve the system
    h = linalg.solve(A, b)

    # Store head and time
    head_history.append(h.copy())
    time_history.append(t)

    # Update h_old
    h_old = h.copy()

    # Increment time
    t += delt

# Convert head_history to a NumPy array
head_history = np.array(head_history)
time_history = np.array(time_history)

# Plot the last time step
plt.figure(figsize=(8, 6))
plt.tricontourf(nodes[:, 0], nodes[:, 1], head_history[-1], 20, cmap='viridis')
plt.colorbar(label='Head (m)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title(f'Head Distribution at t = {time_history[-1]:.0f} days')
plt.show()


# Example: Plotting head at a specific node over time:
node_index = 100  # Example node index
plt.figure(figsize=(8, 6))
plt.plot(time_history, head_history[:, node_index])
plt.xlabel('Time (days)')
plt.ylabel('Head (m)')
plt.title(f'Head at Node {node_index} over Time')
plt.grid(True)
plt.show()


# Example: Animating the head distribution over time (if you have ffmpeg installed)
# fig, ax = plt.subplots()
# cont = ax.tricontourf(nodes[:, 0], nodes[:, 1], head_history[0], 20, cmap='viridis') # Initial frame

# def animate(i):
#     cont.remove()  # Clear the previous frame
#     cont = ax.tricontourf(nodes[:, 0], nodes[:, 1], head_history[i], 20, cmap='viridis')
#     ax.set_title(f'Head Distribution at t = {time_history[i]:.0f} days')
#     return cont,

# ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(head_history), interval=50, blit=False)
# ani.save('head_animation.gif', writer='pillow', fps=10) # Save as GIF (requires imageio)
# plt.show()
