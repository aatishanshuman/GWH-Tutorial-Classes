import numpy as np
import matplotlib.pyplot as plt
import flopy
import flopy.utils.binaryfile as bf
from pathlib import Path

# Define workspace and model name
workspace = Path("D:/Courses Taught/Groundwater Hydrology/GWH Tutorial/Flopy/")
name = "tutorial01_mf"

# Create MODFLOW model object
mf = flopy.modflow.Modflow(name, exe_name="mf2005", model_ws=str(workspace))

# Model domain and grid definition
Lx, Ly, ztop, zbot = 200.0, 200.0, 0.0, -50.0
nlay, nrow, ncol = 1, 20,20 
delr, delc = Lx / ncol, Ly / nrow
botm = np.linspace(ztop, zbot, nlay + 1)[1:]

# Define the discretization package (DIS)
dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc, top=ztop, botm=botm)

# Define boundary conditions (BAS6)
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)  # 1: Active, 0: No-flow, -1: Constant Head
ibound[:, :, 0], ibound[:, :, -1] = -1, -1  # Left & right boundaries as constant head

# Initial conditions (starting head)
strt = np.ones((nlay, nrow, ncol), dtype=np.float32) * 5  # Set an initial head
strt[:, :, 0], strt[:, :, -1] = 100, 90  # Fixed heads

bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

# Define hydraulic properties (LPF)
lpf = flopy.modflow.ModflowLpf(mf, hk=3.33, vka=1)

# Output control (OC)
spd = {(0, 0): ["print head", "print budget", "save head", "save budget"]}
oc = flopy.modflow.ModflowOc(mf, compact=True)

# Solver package (PCG)
pcg = flopy.modflow.ModflowPcg(mf)

# Write MODFLOW input files
mf.write_input()

# Run MODFLOW
success, buff = mf.run_model(silent=False, report=True)
if not success:
    raise RuntimeError("MODFLOW did not terminate normally.")

def plot_results():
    """Extracts and plots head distribution (contours) and velocity vectors (quiver)."""
    # Read head file
    hds = bf.HeadFile(workspace / f"{name}.hds")
    times = hds.get_times()
    head = hds.get_data(totim=times[-1])

    # Read cell budget file
    cbb = bf.CellBudgetFile(workspace / f"{name}.cbc")
    frf = cbb.get_data(text="FLOW RIGHT FACE", totim=times[-1])[0]
    fff = cbb.get_data(text="FLOW FRONT FACE", totim=times[-1])[0]

    # Compute specific discharge (velocity vectors)
    qx, qy, _ = flopy.utils.postprocessing.get_specific_discharge((frf, fff, None), mf, head)

    # Set up subplots: Left → Contour, Right → Quiver
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # --- Left subplot: Contour Plot (Head Distribution) ---
    ax1 = axes[0]
    modelmap1 = flopy.plot.PlotMapView(model=mf, layer=0, ax=ax1)
    contour = modelmap1.contour_array(head, levels=np.linspace(head.min(), head.max(), 11), cmap="viridis")
    fig.colorbar(contour, ax=ax1, label="Head (m)")

    ax1.set_title("Head Distribution (Contour Plot)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    # --- Right subplot: Quiver Plot (Flow Vectors) ---
    ax2 = axes[1]
    modelmap2 = flopy.plot.PlotMapView(model=mf, layer=0, ax=ax2)
    modelmap2.plot_ibound()
    modelmap2.plot_grid()
    quiver = modelmap2.plot_vector(qx, qy, scale=50, color="black")
    ax2.set_title("Flow Vectors (Quiver Plot)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")

    plt.show()


plot_results()