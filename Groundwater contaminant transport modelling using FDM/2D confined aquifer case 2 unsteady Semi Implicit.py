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
delx =100
dely = 100
T = 10
S = 0.0001
Q = 0.05
t_total = 365
delt = 1  # Adjust for stability!
alpha=0.5

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




def gwflow(wx1,wy1,r1, wx2 , wy2, r2):
    
    # Initialize head
    h = np.zeros(numnodes)
    h_old = np.zeros(numnodes)+100
    h_old[left] = 100
    h_old[right] = 100
    

    
    
    
    # Store head values at all time steps
    head_history = []
    time_history = []
    
    # Implicit time-stepping
    t = 0.0
    while t <= t_total:
        print(t)
        A = np.zeros((numnodes, numnodes))
        b = np.zeros(numnodes)
    
        # Interior nodes
        for i in range(numnodes):
            x = nodes[i][0]
            y = nodes[i][1]
    
            # Coefficient for implicit method
            A[i, i] = alpha*(4*T*delt /(S*delx**2)) + 1 # Diagonal term for implicit method
            if i not in boundary:
                # Use closest_node for finding neighbors
                north_node = closest_node([x, y + dely], nodes)
                south_node = closest_node([x, y - dely], nodes)
                east_node = closest_node([x + delx, y], nodes)
                west_node = closest_node([x - delx, y], nodes)
    
                # Set coefficients for neighboring nodes
                A[i, north_node] = -alpha*T*delt / (delx**2*S) 
                A[i, south_node] = -alpha*T*delt / (delx**2*S)
                A[i, east_node] = -alpha*T*delt / (delx**2*S)
                A[i, west_node] = -alpha*T*delt / (delx**2*S)
                
                b[i] = h_old[i] + (1-alpha)*(T*delt /(S*delx**2))* (h_old[north_node] + h_old[south_node] + h_old[east_node] + h_old[west_node] - 4 * h_old[i])
                
            
    
            if i in top:
                south_node = closest_node([x, y - dely], nodes)
                east_node = closest_node([x + delx, y], nodes)
                west_node = closest_node([x - delx, y], nodes)
    
                A[i, south_node] = -2*alpha*T*delt / (delx**2*S)
                A[i, east_node] = -alpha*T*delt / (delx**2*S)
                A[i, west_node] = -alpha*T*delt / (delx**2*S)
                
                b[i] = h_old[i] + (1-alpha)*(T*delt /(S*delx**2))* (2* h_old[south_node] + h_old[east_node] + h_old[west_node] - 4 * h_old[i])
    
            if i in bottom:
                north_node = closest_node([x, y + dely], nodes)
                east_node = closest_node([x + delx, y], nodes)
                west_node = closest_node([x - delx, y], nodes)
    
                A[i, north_node] = -alpha*2*T*delt / (delx**2*S)
                A[i, east_node] = -alpha*T*delt / (delx**2*S)
                A[i, west_node] = -alpha*T*delt / (delx**2*S)
                
                b[i] = h_old[i] + (1-alpha)*(T*delt /(S*delx**2))* (2* h_old[north_node] + h_old[east_node] + h_old[west_node] - 4 * h_old[i])
    
        # Injection well
        b[closest_node([wx1,wy1], nodes)]+=r1*delt/S 
        b[closest_node([wx2, wy2], nodes)]+=r2*delt /S
        
        # Boundary conditions
        b[left] = 100
        A[left, left] = 1
    
        b[right] =100
        A[right, right] = 1
    
        # Solve the system using the implicit method
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
    
    return head_history , time_history



head_history, time_history=gwflow(200, 500,-.03, 500, 1500, 0.05)


#%%

# Plot the last time step
plt.figure(figsize=(8, 6))
plt.tricontourf(nodes[:, 0], nodes[:, 1], head_history[0],20, cmap='viridis')
plt.colorbar(label='Head (m)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title(f'Head Distribution at t = {time_history[-1]:.0f} days')
plt.show()


#%%
# Example: Plotting head at a specific node over time:
node_index = closest_node([1000, 1000], nodes) # Example node index
plt.figure(figsize=(8, 6))
plt.plot(time_history, head_history[:, closest_node([1000, 1000], nodes)])
plt.xlabel('Time (days)')
plt.ylabel('Head (m)')
plt.title(f'Head at Node {node_index} over Time')
plt.grid(True)
plt.show()

#%%

# # Example: Animating the head distribution over time (if you have ffmpeg installed)
# fig, ax = plt.subplots()
# cont = ax.tricontourf(nodes[:, 0], nodes[:, 1], head_history[0], 20, cmap='viridis') # Initial frame

# def animate(i):
    
#     cont = ax.tricontourf(nodes[:, 0], nodes[:, 1], head_history[i], 20, cmap='viridis')
#     ax.set_title(f'Head Distribution at t = {time_history[i]:.0f} days')
#     return cont,

# ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(head_history), interval=50, blit=False)
# ani.save('head_animation.gif', writer='pillow', fps=10) # Save as GIF (requires imageio)
# plt.show()


#%%
# import matplotlib.animation as animation

# # Assume 'nodes' is your list of 3D coordinates and 'head_history' is a list of head values over time
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Initial frame with 'head_history[0]' and 'nodes' having 3D data
# surf = ax.plot_trisurf(nodes[:, 0], nodes[:, 1], head_history[0], cmap='viridis', linewidth=0, antialiased=False)

# # Set up plot limits and labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Head')
# ax.set_title(f'Head Distribution at t = {time_history[0]:.0f} days')

# def animate(i):
#     ax.cla()  # Clear the axis to update the plot for each frame
    
#     # Recreate the surface plot for the current time frame
#     ax.plot_trisurf(nodes[:, 0], nodes[:, 1], head_history[i], cmap='viridis', linewidth=0, antialiased=False)
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Head')
#     ax.set_title(f'Head Distribution at t = {time_history[i]:.0f} days')
    
#     return surf,

# ani = animation.FuncAnimation(fig, animate, frames=len(head_history), interval=50, blit=False)
# ani.save('head_animation_3d.gif', writer='pillow', fps=10)  # Save as GIF (requires imageio)
# plt.show()
