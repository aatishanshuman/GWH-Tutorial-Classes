import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis("off")

# Box and arrow styles
box_style = dict(boxstyle="round,pad=0.5", fc="#D6EAF8", ec="black", lw=1.2)
arrow_style = dict(arrowstyle="->", color="black", lw=1.2)

# Define steps with positions

# Define steps with positions and adjusted spacings
steps = {
    "start": ("Start", (0.5, 0.90)),
    "import": ("Import libraries and patmodels", (0.5, 0.82)),
    "input": ("Load input files (ibound, strt, icbund)", (0.5, 0.74)),
    "init": ("Load initial head & conc using pat_initalcondn", (0.5, 0.66)),
    "clean": ("Replace missing values with NaN", (0.5, 0.58)),
    "plot1": ("Plot initial head and concentration", (0.5, 0.50)),
    "model_run": ("Run pat_model with sample pumping for Manual Optimization", (0.5, 0.42)),
    "extract": ("Extract conc. at monitoring wells", (0.5, 0.34)),
    "plot2": ("Plot final concentration map", (0.5, 0.26)),
    "bounds": ("Define pumping bounds", (0.5, 0.18)),
    "pso": ("Run PSO optimization", (0.5, 0.10)),
    "result": ("Display best pumping strategy", (0.5, 0.02)),
    "end": ("End", (0.5, -0.06))
}


# Draw boxes
for key, (text, pos) in steps.items():
    ax.text(pos[0], pos[1], text, ha="center", va="center", bbox=box_style, fontsize=12,fontfamily='Arial')

# Draw arrows between steps
keys = list(steps.keys())
for i in range(len(keys) - 1):
    x0, y0 = steps[keys[i]][1]
    x1, y1 = steps[keys[i + 1]][1]
    ax.annotate("", xy=(x1, y1 + 0.025), xytext=(x0, y0 - 0.025), arrowprops=arrow_style)

plt.title("Flowchart: Pump and Treat Optimization Workflow", fontsize=12, weight="bold")
plt.tight_layout()
plt.show()
