import os
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import flopy

# Define executable names for MODFLOW and MT3DMS
exe_name_mf = "mf2005"
exe_name_mt = "mt3dms"

# Create a temporary working directory
temp_dir = TemporaryDirectory()
workdir = temp_dir.name

# Model setup
dirname, mixelm = "p03", 3
model_ws = os.path.join(workdir, dirname)

# Model domain and parameters
nlay, nrow, ncol = 1, 21, 21
delr, delc, delv = 10, 10, 10  # Grid spacing (m)
Lx = (ncol - 1) * delr  # Model length in x-direction

# Hydrogeological parameters
v, prsity, al, trpt = 1.0 / 3.0, 0.3, 10.0, 0.3
q, q0, c0 = v * prsity, 1.0, 1000.0  # Flow and concentration parameters
perlen_mf, perlen_mt = 365.0 * 5, 365 * 5.0  # Simulation period (days)
hk, laytyp = 1.0, 0  # Hydraulic conductivity and layer type

# -------------------------
# Create and configure MODFLOW model
# -------------------------
mf = flopy.modflow.Modflow(
    modelname=f"{dirname}_mf", model_ws=model_ws, exe_name=exe_name_mf
)

# Discretization package
dis = flopy.modflow.ModflowDis(
    mf, nlay, nrow, ncol, delr=delr, delc=delc, top=0.0, botm=[-delv], perlen=perlen_mf
)

# Boundary conditions
ibound = np.ones((nlay, nrow, ncol), dtype=int)
ibound[0, :, [0, -1]] = -1  # Constant head boundaries

# Initial conditions
strt = np.zeros((nlay, nrow, ncol), dtype=float)
strt[0, :, 0] = q * Lx  # Initial head at left boundary

# Add MODFLOW packages
flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
flopy.modflow.ModflowLpf(mf, hk=hk, laytyp=laytyp)
flopy.modflow.ModflowWel(mf, stress_period_data=[[0, 15, 15, q0]])
flopy.modflow.ModflowPcg(mf)
flopy.modflow.ModflowLmt(mf)

# Write input files and run MODFLOW
mf.write_input()
mf.run_model(silent=True)

# -------------------------
# Create and configure MT3DMS model
# -------------------------
mt = flopy.mt3d.Mt3dms(
    modelname=f"{dirname}_mt",
    model_ws=model_ws,
    exe_name=exe_name_mt,
    modflowmodel=mf,
)

# Basic transport package
flopy.mt3d.Mt3dBtn(mt, icbund=1, prsity=prsity, sconc=0)

# Advection package
adv = flopy.mt3d.Mt3dAdv(
    mt,
    mixelm=mixelm,
    dceps=1.0e-5,
    nplane=1,
    npl=0,
    nph=16,
    npmin=2,
    npmax=32,
    nlsink=1,
    npsink=16,
    percel=0.5,
)

# Dispersion and source-sink packages
flopy.mt3d.Mt3dDsp(mt, al=al, trpt=trpt)
flopy.mt3d.Mt3dSsm(mt, stress_period_data={0: [0, 15, 15, c0, 2]})
flopy.mt3d.Mt3dGcg(mt)

# Write input files and run MT3DMS
mt.write_input()
mt.run_model(silent=False)

# -------------------------
# Read and visualize results
# -------------------------
# Read concentration data
ucnobj = flopy.utils.UcnFile(os.path.join(model_ws, "MT3D001.UCN"))
conc = ucnobj.get_alldata()[0]  # Get first time step concentration


def plot_results(mf, conc):
    """
    Plot concentration contour map.

    Parameters:
    - mf (Modflow): MODFLOW model instance.
    - conc (ndarray): Concentration data.
    - title (str): Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    pmv = flopy.plot.PlotMapView(model=mf, ax=ax)

    # Plot grid and concentration contours
    pmv.plot_grid(color=".5", alpha=0.2)
    cs = pmv.contour_array(conc, levels=np.linspace(0, 50, 10), cmap="jet", plot_triplot=True)

    plt.clabel(cs)
    plt.xlabel("Distance along X-axis (m)")
    plt.ylabel("Distance along Y-axis (m)")
    plt.show()


# Plot results for the HMOC advection scheme
plot_results(mf, conc)

