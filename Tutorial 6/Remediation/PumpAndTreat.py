import numpy as np
import matplotlib.pyplot as plt
from patmodels import *

head,conc,X,Y=pat_initalcondn(ibound_path, icbund_path, strt_path)


# Replace no-data values with NaN for plotting
head = np.where(head == -999.99, np.nan, head)
conc = np.where(conc == 1e+30, np.nan, conc)

# Set up the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(wspace=0.3)

# --- Plot 1: Hydraulic Head ---
c1 = ax1.contourf(X, Y, head, levels=np.linspace(np.nanmin(head), np.nanmax(head), 10), cmap="viridis")
plt.colorbar(c1, ax=ax1, label="Head (m)")
ax1.set_title("Hydraulic Head")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")

# Plot wells on head plot
for well_group in [iwell_xy, monitoring_wells_xy]:
    for name, r, c in well_group:
        ax1.plot(r, c, 'ro')
        ax1.text(r, c, name, color='white', fontsize=8)

# --- Plot 2: Contaminant Concentration ---
c2 = ax2.contourf(X, Y, conc, levels=np.linspace(np.nanmin(conc), np.nanmax(conc), 10), cmap="inferno")
plt.colorbar(c2, ax=ax2, label="Concentration (ppm)")
ax2.set_title("Contaminant Concentration")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")

# Plot wells on concentration plot
for name, r, c in monitoring_wells_xy:
    ax2.plot(r, c, 'ro')
    ax2.text(r, c, name, color='white', fontsize=8)

# Final layout
plt.tight_layout()
plt.show()


#%%

conc_=pat_model(-1000,1000,1000)
conc_[conc_ == 1e+30] = np.nan

# Extract concentrations at monitoring well locations
conc_vals = []
for _, r, c in monitoring_wells:
    if 0 <= r < conc_.shape[0] and 0 <= c < conc_.shape[1]:
        value = conc_[r, c]
        if np.isnan(value):
            value = 0  # or maybe set to 1e30 to harshly penalize
        conc_vals.append(value)
    else:
        conc_vals.append(0)  # Out of bounds fallback     
    print(_,value)



fig, ax = plt.subplots(figsize=(8, 3))
cs = ax.contourf(X, Y, conc_, levels=np.linspace(0, np.nanmax(conc_), 10), cmap="jet")
cbar = plt.colorbar(cs, ax=ax)
cbar.set_label("Concentration (ppm)")

for well_group in [iwell_xy, wells_xy, monitoring_wells_xy]:
    for name, r, c in well_group:
        ax.plot(r, c, 'ro')
        ax.text(r, c, name, color='white', fontsize=8)

ax.set_title("Contaminant Concentration at Final Time Step")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
plt.tight_layout()
plt.show()


#%%


pump_treat_objective([0,0,0])

#%%
# Define bounds (e.g., between -200 and -10 for pumping rates)
lb = np.array([-200,-1000,0])
ub = np.array([0,0,500])

best_pumping, final_cost, history = pso_pat(pump_treat_objective, lb, ub, max_iter=10)
print(f"\nBest Pumping Strategy: Q1 = {best_pumping[0]:.2f}, Q2 = {best_pumping[1]:.2f}, Q3 = {best_pumping[2]:.2f}")
print(f"Final Cost: {final_cost:.2f}")




