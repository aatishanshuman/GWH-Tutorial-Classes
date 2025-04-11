import numpy as np
import pandas as pd
import flopy
import flopy.utils.binaryfile as bf
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------- #
# 1. MODFLOW Simulation Function
# ---------------------------- #
def problem1(hk: float, workspace: str, model_name: str):
    """
    Run MODFLOW model with specified hydraulic conductivity and return head distribution.

    Parameters:
        hk (float): Hydraulic conductivity value.
        workspace (str): Directory for model files.
        model_name (str): MODFLOW model name.

    Returns:
        X, Y: Meshgrid arrays.
        head: Simulated head (2D array, layer 0).
    """
    # Create MODFLOW model object
    mf = flopy.modflow.Modflow(model_name, exe_name="mf2005", model_ws=workspace)

    # Model domain
    Lx, Ly, ztop, zbot = 200.0, 200.0, 0.0, -10.0
    nlay, nrow, ncol = 1, 20, 20
    delr, delc = Lx / ncol, Ly / nrow
    botm = np.linspace(ztop, zbot, nlay + 1)[1:]

    # Discretization
    flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc, top=ztop, botm=botm)

    # Boundary conditions
    ibound = np.expand_dims(pd.read_csv('D:/Courses Taught/Groundwater Hydrology/GWH Tutorial/Inverse Modelling/iboundProblem1.csv', header=None).values, axis=0)
    strt = np.expand_dims(pd.read_csv('D:/Courses Taught/Groundwater Hydrology/GWH Tutorial/Inverse Modelling/startingheadsProblem1.csv', header=None).values, axis=0)
    flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

    # Hydraulic properties
    flopy.modflow.ModflowLpf(mf, hk=hk, vka=1)

    # Well
    well_location = [0, 10, 10]
    flopy.modflow.ModflowWel(mf, stress_period_data={0: [[*well_location, -500]]})

    # Output control & solver
    flopy.modflow.ModflowOc(mf, compact=True)
    flopy.modflow.ModflowPcg(mf)

    # Write and run model
    mf.write_input()
    success, _ = mf.run_model(silent=True)
    if not success:
        raise RuntimeError("MODFLOW did not terminate normally.")

    # Read head results
    hds = bf.HeadFile(Path(workspace) / f"{model_name}.hds")
    head = hds.get_data(totim=hds.get_times()[-1])[0]
    head[head == -999.99] = np.nan

    # Create meshgrid
    x = np.linspace(0, Lx, ncol)
    y = np.linspace(0, Ly, nrow)
    X, Y = np.meshgrid(y, x)

    return X, Y, np.flip(head, axis=0)

# ---------------------------- #
# 2. Setup Synthetic Observations
# ---------------------------- #
np.random.seed(40)

workspace = "D:/Courses Taught/Groundwater Hydrology/GWH Tutorial/Flopy/"
model_name = "tutorial01_mf"
true_hk = 3.33

X, Y, head_true = problem1(hk=true_hk, workspace=workspace, model_name=model_name)

# Select valid observation points
n_obs = 10
obs_locs = []
while len(obs_locs) < n_obs:
    row, col = np.random.randint(0, 20), np.random.randint(0, 20)
    if not np.isnan(head_true[row, col]) and (row, col) not in obs_locs:
        obs_locs.append((row, col))

obs_locs = np.array(obs_locs)
obs_heads = [head_true[row, col] for row, col in obs_locs]

# ---------------------------- #
# 3. Visualize Head + Observation Points
# ---------------------------- #
delr = delc = 200 / 20
x_coords = [col * delr + delr / 2 for row, col in obs_locs]
y_coords = [row * delc + delc / 2 for row, col in obs_locs]

plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, head_true, levels=20, cmap='viridis')
plt.colorbar(cp, label='Head (m)')
plt.scatter(x_coords, y_coords, color='red', edgecolor='black', s=40, label='Observation Points')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Head Distribution with Observation Points')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------- #
# 4. Particle Swarm Optimization (PSO)
# ---------------------------- #
def pso(objective, lb, ub, num_particles=20, max_iter=10, w=0.5, c1=1.5, c2=1.5):
    positions = np.random.uniform(lb, ub, size=(num_particles, 1))
    velocities = np.zeros_like(positions)

    p_best_pos = positions.copy()
    p_best_val = np.array([objective(pos) for pos in positions])

    g_best_index = np.argmin(p_best_val)
    g_best_pos = p_best_pos[g_best_index].copy()
    g_best_val = p_best_val[g_best_index]

    rmse_history = [g_best_val]  # Track RMSE over iterations

    for i in range(max_iter):
        for j in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[j] = (
                w * velocities[j]
                + c1 * r1 * (p_best_pos[j] - positions[j])
                + c2 * r2 * (g_best_pos - positions[j])
            )
            positions[j] += velocities[j]
            positions[j] = np.clip(positions[j], lb, ub)

            score = objective(positions[j])

            if score < p_best_val[j]:
                p_best_val[j] = score
                p_best_pos[j] = positions[j]

                if score < g_best_val:
                    g_best_val = score
                    g_best_pos = positions[j]

        rmse_history.append(g_best_val)
        print(f"Iteration {i+1}: Best RMSE = {g_best_val:.4f}, hk = {g_best_pos[0]:.4f}")

    return g_best_pos[0], g_best_val, rmse_history

# ---------------------------- #
# 5. Objective Function (RMSE)
# ---------------------------- #
def objective(hk):
    _, _, head_sim = problem1(hk[0], workspace=workspace, model_name=model_name)
    sim_heads = [head_sim[row, col] for row, col in obs_locs]
    return np.sqrt(np.nanmean((np.array(sim_heads) - np.array(obs_heads)) ** 2))

# ---------------------------- #
# 6. Run PSO
# ---------------------------- #
hk_opt, fopt, rmse_history = pso(objective, lb=0.1, ub=10.0, num_particles=20, max_iter=10)

# Plot RMSE vs Iteration
plt.figure(figsize=(8, 5))
plt.plot(rmse_history, marker='o', linestyle='-', color='blue')
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("RMSE vs. Iteration")
plt.grid(True)
plt.tight_layout()
plt.show()


print(f"\n✅ Optimal hk: {hk_opt:.4f}")
print(f"✅ Final RMSE: {fopt:.4f}")



# Get simulated heads using optimized hk
_, _, head_opt = problem1(hk=hk_opt, workspace=workspace, model_name=model_name)

# Extract simulated heads at observation points
simulated_heads = [head_opt[row, col] for row, col in obs_locs]

#%%
# Plot bar chart
plt.figure(figsize=(10, 5))
bar_width = 0.35
indices = np.arange(len(obs_heads))

plt.bar(indices, obs_heads, bar_width, label="Observed Heads", color='green')
plt.bar(indices + bar_width, simulated_heads, bar_width, label="Simulated Heads (Optimized)", color='orange')
plt.ylim(60,100)


plt.xlabel("Observation Point Index")
plt.ylabel("Head (m)")
plt.title("Observed vs. Simulated Head at Observation Points")
plt.xticks(indices + bar_width / 2, labels=[f"P{i+1}" for i in range(len(obs_heads))])
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()
