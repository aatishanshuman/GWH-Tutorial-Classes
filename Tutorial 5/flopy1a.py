# Import necessary libraries
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import flopy
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf

# --------------------------
#  Model Setup
# --------------------------
# Define model domain
Lx, Ly = 200,200
ztop, zbot = 30,0
nlay, nrow, ncol = 1, 20, 20
delr, delc = Lx / ncol, Ly / nrow
botm = np.linspace(ztop, zbot, nlay + 1)

# Define hydraulic properties
hk, vka = 3.33, 1.0
sy, ss = 0.1, 1.0e-4
laytyp = 1

# Define initial and boundary conditions
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)  # All active
strt = np.full((nlay, nrow, ncol), 10.0, dtype=np.float32)  # Initial head

# Define stress periods
nper = 1
perlen = [1]
nstp = [1]
steady = [True]

# Create workspace
temp_dir = TemporaryDirectory()
workspace = Path(temp_dir.name)
name = "tutorial02_mf"

# Initialize MODFLOW model
mf = flopy.modflow.Modflow(name, exe_name="mf2005", model_ws=str(workspace))
dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc, 
                               top=ztop, botm=botm[1:], nper=nper, perlen=perlen, 
                               nstp=nstp, steady=steady)
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vka, sy=sy, ss=ss, laytyp=laytyp, ipakcb=53)
pcg = flopy.modflow.ModflowPcg(mf)

# --------------------------
#  General Head Boundary (GHB)
# --------------------------
def create_ghb(stage_left, stage_right):
    """Creates general head boundary conditions for a given left & right stage."""
    bound = []
    cond_left, cond_right = hk * (stage_left - zbot) * delc, hk * (stage_right - zbot) * delc
    for il in range(nlay):
        for ir in range(nrow):
            bound.append([il, ir, 0, stage_left, cond_left])  # Left boundary
            bound.append([il, ir, ncol - 1, stage_right, cond_right])  # Right boundary
    return bound

# GHB for each stress period
ghb_sp1 = create_ghb(100, 80.0)
stress_period_data = {0: ghb_sp1}
ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=stress_period_data)

# --------------------------
#  Well Package
# --------------------------
pumping_rate = -500.0
well_location1 = [0,8,10]
well_location2 = [0,10,6]
wel_sp1 = [[*well_location1, pumping_rate], [*well_location2, 800] ]

stress_period_data = {0: wel_sp1}
wel = flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)

# --------------------------
#  Output Control
# --------------------------
stress_period_data = {(kper, kstp): ["save head", "save drawdown", "save budget", "print head", "print budget"]
                      for kper in range(nper) for kstp in range(nstp[kper])}
oc = flopy.modflow.ModflowOc(mf, stress_period_data=stress_period_data, compact=True)

# --------------------------
#  Run Model
# --------------------------
mf.write_input()
success, mfoutput = mf.run_model(silent=True, pause=False)
assert success, "MODFLOW did not terminate normally."

# --------------------------
# Extract Results
# --------------------------
headobj = bf.HeadFile(workspace / f"{name}.hds")
times = headobj.get_times()
cbb = bf.CellBudgetFile(workspace / f"{name}.cbc")

# --------------------------
#  Visualization
# --------------------------
def plot_results():
    """Plots head distribution and flow vectors for different stress periods."""
    mytimes = [1.0]
    levels = np.linspace(0, 10, 11)

    fig, axes = plt.subplots(len(mytimes), 1, figsize=(8,4), constrained_layout=True)

   
        
    time=1
    
    head = headobj.get_data(totim=time)
    frf = cbb.get_data(text="FLOW RIGHT FACE", totim=time)[0]
    fff = cbb.get_data(text="FLOW FRONT FACE", totim=time)[0]

    ax = plt.subplot(111)
    pmv = flopy.plot.PlotMapView(model=mf, layer=0, ax=ax)
    pmv.plot_ibound()
    pmv.plot_bc("GHB", alpha=0.5)
    pmv.plot_grid()
    
    if head.min() != head.max():
        cs = pmv.contour_array(head, levels=levels, cmap="viridis")
        plt.clabel(cs, inline=1, fontsize=10, fmt="%1.1f")
        pmv.plot_vector(frf, fff, color="black")

    ax.set_title("Velocity Directions")

    plt.show()


# --------------------------
#  Plot Results
# --------------------------
plot_results()
