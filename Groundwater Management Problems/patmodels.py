import os
import numpy as np
import matplotlib.pyplot as plt
import flopy
from tempfile import TemporaryDirectory
import pandas as pd

#%%


exe_name_mf = "mf2005"
exe_name_mt = "mt3dms"

# --- Constants ---
Lx, Ly, ztop, zbot = 4000.0, 2000.0, 0.0, -10.0
nlay, nrow, ncol = 1, 20,40
delr, delc = Lx / ncol, Ly / nrow
botm = np.linspace(ztop, zbot, nlay + 1)[1:]
hk, vka, prsity = 75, 0.5, 0.35
perlen = 365*10  #pollution time
perlen2 = 365*2  #remediation time
q0, c0 = 10.0, 1000.0
nper, nstp = 1, 1
al,trpt=20,0.5


wells_xy = [
    ('W1', 1000, 1000),
    ('W2', 2000, 1000),
    ('W3', 1750, 1250)
]

monitoring_wells_xy = [
    ('MW1', 900, 950),
    ('MW2', 1500, 750),
    ('MW3', 1750, 1200),
    ('MW4', 2000, 950),
    ('MW5', 2500, 1000),
    ('MW6', 3000,1000)
]

iwell_xy=[('IWELL1', 500,1250)]

def get_row_col_from_xy(x, y, delr, delc, nrow):
    col = round(x / delr)
    row = round((Ly - y) / delc)
    
    # Clamp row and col to stay within model domain bounds
    row = max(0, min(nrow - 1, row))
    col = max(0, min(int(Lx / delr) - 1, col))
    
    return row, col



iwell=[(name, *get_row_col_from_xy(x, y, delr, delc, nrow)) for name, x, y in iwell_xy]
wells= [(name, *get_row_col_from_xy(x, y, delr, delc, nrow)) for name, x, y in wells_xy]
monitoring_wells = [(name, *get_row_col_from_xy(x, y, delr, delc, nrow)) for name, x, y in monitoring_wells_xy]




def pat_initalcondn(ibound_path, icbund_path, strt_path, for_ic=False):
    
    temp_dir = TemporaryDirectory()
    model_ws = os.path.join(temp_dir.name, "remediation_model")
    os.makedirs(model_ws, exist_ok=True)

    ibound = np.expand_dims(pd.read_csv(ibound_path, header=None).values, axis=0)
    strt = np.expand_dims(pd.read_csv(strt_path, header=None).values, axis=0)
    icbund = np.expand_dims(pd.read_csv(icbund_path, header=None).values, axis=0)

    mf = flopy.modflow.Modflow("remediation", exe_name=exe_name_mf, model_ws=model_ws)
    flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                             botm=botm, nper=nper, perlen=perlen, nstp=nstp, steady=False)
    flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    flopy.modflow.ModflowLpf(mf, hk=hk, vka=vka)
    flopy.modflow.ModflowPcg(mf)
    flopy.modflow.ModflowOc(mf, stress_period_data={(0, 0): ["save head", "print head", "save budget"]}, compact=True)
    flopy.modflow.ModflowWel(mf, stress_period_data={0: [[0, iwell[0][1], iwell[0][2], q0]]})
    flopy.modflow.ModflowLmt(mf, output_file_name="mt3d_link.ftl")

    mf.write_input()
    mf.run_model(silent=True)

    mt = flopy.mt3d.Mt3dms("remediation_mt", model_ws=model_ws, exe_name=exe_name_mt, modflowmodel=mf)
    timprs = np.linspace(0, perlen, nstp)
    flopy.mt3d.Mt3dBtn(mt, icbund=icbund, prsity=prsity, sconc=np.zeros((nlay, nrow, ncol)),
                       nprs=len(timprs), timprs=timprs)
    flopy.mt3d.Mt3dAdv(mt, mixelm=3, dceps=1e-5, nplane=1, npl=0, nph=16,
                       npmin=2, npmax=32, nlsink=1, npsink=16, percel=0.5)
    flopy.mt3d.Mt3dDsp(mt, al=al, trpt=trpt)
    flopy.mt3d.Mt3dSsm(mt, stress_period_data={0: [0, iwell[0][1], iwell[0][2], c0, 2]})
    flopy.mt3d.Mt3dGcg(mt)

    mt.write_input()
    mt.run_model(silent=True)

    head = flopy.utils.HeadFile(os.path.join(model_ws, "remediation.hds")).get_data()[-1]
    conc = flopy.utils.UcnFile(os.path.join(model_ws, "MT3D001.UCN")).get_data()[-1]

    x = np.linspace(0, Lx, ncol)
    y = np.linspace(0, Ly, nrow)[::-1]
    X, Y = np.meshgrid(x,y)

    return (head, conc) if for_ic else (np.flip(head, 0), np.flip(conc, 0), X, Y)


def pat_model(Q1, Q2, Q3, ibound_path=None, icbund_path=None, strt_path=None):
    strt, sconc = pat_initalcondn(ibound_path, icbund_path, strt_path, for_ic=True)

    temp_dir = TemporaryDirectory()
    model_ws = os.path.join(temp_dir.name, "remediation_model")
    os.makedirs(model_ws, exist_ok=True)

    ibound = np.expand_dims(pd.read_csv(ibound_path, header=None).values, axis=0)
    strt = np.expand_dims(strt, axis=0)
    sconc = np.expand_dims(sconc, axis=0)

    mf = flopy.modflow.Modflow("remediation", exe_name=exe_name_mf, model_ws=model_ws)
    flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                             botm=botm, nper=nper, perlen=perlen2, nstp=nstp, steady=False)
    flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    flopy.modflow.ModflowLpf(mf, hk=hk, vka=vka)
    flopy.modflow.ModflowPcg(mf)
    flopy.modflow.ModflowOc(mf, stress_period_data={(0, 0): ["save head", "print head", "save budget"]}, compact=True)
    flopy.modflow.ModflowWel(mf, stress_period_data={0: [
        [0, wells[0][1], wells[0][2], Q1],
        [0, wells[1][1], wells[1][2], Q2],
        [0, wells[2][1], wells[2][2], Q3]
    ]})
    flopy.modflow.ModflowLmt(mf, output_file_name="mt3d_link.ftl")

    mf.write_input()
    mf.run_model(silent=True)

    mt = flopy.mt3d.Mt3dms("remediation_mt", model_ws=model_ws, exe_name=exe_name_mt, modflowmodel=mf)
    timprs = np.linspace(0, perlen2, nstp)
    flopy.mt3d.Mt3dBtn(mt, icbund=ibound, prsity=prsity, sconc=sconc,
                       nprs=len(timprs), timprs=timprs)
    flopy.mt3d.Mt3dAdv(mt, mixelm=3, dceps=1e-5, nplane=1, npl=0, nph=16,
                       npmin=2, npmax=32, nlsink=1, npsink=16, percel=0.5)
    flopy.mt3d.Mt3dDsp(mt, al=al, trpt=trpt)

    # Source-Sink Mixing - no new concentration added in this step
    flopy.mt3d.Mt3dSsm(mt, stress_period_data={0: [
        [0, wells[0][1], wells[0][2], 0,2],
        [0, wells[1][1], wells[1][2], 0,2],
        [0, wells[2][1], wells[2][2], 0,2]
    ]})
    flopy.mt3d.Mt3dGcg(mt)

    mt.write_input()
    mt.run_model(silent=True)

    conc = flopy.utils.UcnFile(os.path.join(model_ws, "MT3D001.UCN")).get_data()[-1]

    return np.flip(conc, 0)


def pump_treat_objective(Q, ibound_path=None, icbund_path=None, strt_path=None,
                         c1=100.0, c2=100.0, c3=100.0 , treatment_threshold=3):
    Q1, Q2, Q3 = Q

    # Run the model with the given pumping rates
    conc = pat_model(Q1, Q2, Q3, ibound_path, icbund_path, strt_path)

    # Extract concentrations at monitoring well locations
    conc_vals = []
    for _, r, c in monitoring_wells:
        if 0 <= r < conc.shape[0] and 0 <= c < conc.shape[1]:
            value = conc[r, c]
            if np.isnan(value):
                value = 0
            conc_vals.append(value)
        else:
            conc_vals.append(0)

    # --- Objective Components ---
    penalty_weight = 1e4
    penalty = sum([(max(0, c - treatment_threshold))**2 for c in conc_vals]) * penalty_weight

    # ✅ Weighted pumping cost using realistic coefficients
    pumping_cost = c1 * abs(Q1) + c2 * abs(Q2) + c3 * abs(Q3)

    total_cost = pumping_cost + penalty

    return total_cost



def pso_pat(objective, lb, ub, num_particles=10, max_iter=10, w=0.5, c1=1.5, c2=1.5):
    positions = np.random.uniform(lb, ub, size=(num_particles, 3))
    velocities = np.zeros_like(positions)

    p_best_pos = positions.copy()
    p_best_val = np.array([objective(pos) for pos in positions])

    g_best_index = np.argmin(p_best_val)
    g_best_pos = p_best_pos[g_best_index].copy()
    g_best_val = p_best_val[g_best_index]

    cost_history = [g_best_val]  # Track cost (objective) over iterations

    print("Starting PSO optimization...\n")

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

        cost_history.append(g_best_val)
        print(f"Iteration {i+1}/{max_iter} | Best Cost = {g_best_val:.4f} | Q = {g_best_pos}")

    print("\nOptimization complete.")
    return g_best_pos, g_best_val, cost_history

