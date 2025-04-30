import numpy as np
import matplotlib.pyplot as plt
from patmodels import *


ibound_path='iboundProblem2.csv'
strt_path='startingheadsProblem2.csv'
icbund_path='icbundProblem2.csv'


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

conc_=pat_model(-1000,1000,1000,ibound_path, icbund_path, strt_path)
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


lb = np.array([-5000,-5000,-5000])
ub = np.array([1000,1000,1000])


best_pumping, final_cost, history = pso_pat(
    lambda Q: pump_treat_objective(Q, ibound_path, icbund_path, strt_path),
    lb, ub, max_iter=50
)

print(f"\nBest Pumping Strategy: Q1 = {best_pumping[0]:.2f}, Q2 = {best_pumping[1]:.2f}, Q3 = {best_pumping[2]:.2f}")
print(f"Final Cost: {final_cost:.2f}")


#%%

# Define lower and upper bounds for the pumping rates
# lb = np.array([-5000, -5000, -5000])
# ub = np.array([1000, 1000, 1000])

# # Number of random combinations to generate
# num_combinations = 15

# # Initialize lists to store the pumping rates and pumping costs
# pumping_costs = []
# pumping_rates = []

# np.random.seed(10)
# # Generate 15 random combinations of pumping rates
# for _ in range(num_combinations):
#     Q = np.random.randint(lb, ub)*100  # Random pumping rates between lb and ub
    
#     # Calculate the pumping cost for the current combination
#     pumping_cost = pump_treat_objective(Q, ibound_path, icbund_path, strt_path)
    
#     # Save the pumping cost and the corresponding pumping rates
#     pumping_costs.append(pumping_cost)
#     pumping_rates.append(Q)

# # Convert the results to a DataFrame
# df = pd.DataFrame(pumping_rates, columns=['Q1', 'Q2', 'Q3'])
# df['Pumping Cost'] = pumping_costs

# # Print the DataFrame
# print(df)
# df.to_csv('pumping_costs.csv', index=False)


#%%


conc_=pat_model(best_pumping[0],best_pumping[1],best_pumping[2],ibound_path, icbund_path, strt_path)
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



import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 4))  # Slightly wider for better aspect
cs = ax.contourf(X, Y, conc_, levels=np.linspace(0, np.nanmax(conc_), 20), cmap="Reds_r")  # smoother colors

# Add colorbar
cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label("Concentration (ppm)", fontsize=10)
cbar.ax.tick_params(labelsize=8)

# Plot well groups with distinct marker styles
for well_group, color, marker in zip(
    [iwell_xy, wells_xy, monitoring_wells_xy],
    ['red', 'dodgerblue', 'orange'],
    ['o', 's', '^']
):
    for name, r, c in well_group:
        ax.plot(r, c, marker, color=color, markersize=10, label=name)
        ax.text(r-10, c-5, f"{name}", color='w', fontsize=10, ha='center', va='center')



# Avoid duplicate legend entries
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.55,1.1),
          fontsize=8, frameon=False,ncols=5)

ax.set_xlabel("X (m)", fontsize=10)
ax.set_ylabel("Y (m)", fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=8)

ax.spines[['top', 'right']].set_visible(False)

# # Add grid
# ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

#%%

import numpy as np
from scipy.optimize import minimize_scalar

# Given concentrations
concs = np.array(conc_vals)
Q=np.array([0.16,1498,928])
# The target sum based on penalty
target_sum = (379659-100*Q.sum()) /10**4

# Function to calculate the sum of squared differences
def penalty_sum(T):
    return np.sum(np.maximum(0, concs - T)**2)

# Minimize the difference between the calculated and target penalty sum
result = minimize_scalar(lambda T: abs(penalty_sum(T) - target_sum), bounds=(0, 5), method='bounded')

# Estimated treatment threshold
T_est = result.x

print(f"✅ Estimated Treatment Threshold (T): {T_est:.4f} mg/L")


#%%
case1=pump_treat_objective(Q, ibound_path, icbund_path, strt_path, treatment_threshold=1)
case2=pump_treat_objective(Q, ibound_path, icbund_path, strt_path, treatment_threshold=0.25)

print((case2-case1)*100/case1, '%')
