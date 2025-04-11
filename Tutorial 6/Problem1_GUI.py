import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import flopy.utils.binaryfile as bf
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QLabel,
    QLineEdit, QProgressBar, QHBoxLayout, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# ------------------------------ #
# 1. Problem 1 - MODFLOW Runner
# ------------------------------ #
def problem1(hk: float, workspace: str, model_name: str):
    mf = flopy.modflow.Modflow(model_name, exe_name="mf2005", model_ws=workspace)

    Lx, Ly, ztop, zbot = 200.0, 200.0, 0.0, -10.0
    nlay, nrow, ncol = 1, 20, 20
    delr, delc = Lx / ncol, Ly / nrow
    botm = np.linspace(ztop, zbot, nlay + 1)[1:]

    flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc, top=ztop, botm=botm)
    ibound = np.expand_dims(pd.read_csv('D:/Courses Taught/Groundwater Hydrology/GWH Tutorial/Inverse Modelling/iboundProblem1.csv', header=None).values, axis=0)
    strt = np.expand_dims(pd.read_csv('D:/Courses Taught/Groundwater Hydrology/GWH Tutorial/Inverse Modelling/startingheadsProblem1.csv', header=None).values, axis=0)
    flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    flopy.modflow.ModflowLpf(mf, hk=hk, vka=1)

    well_location = [0, 10, 10]
    flopy.modflow.ModflowWel(mf, stress_period_data={0: [[*well_location, -500]]})
    flopy.modflow.ModflowOc(mf, compact=True)
    flopy.modflow.ModflowPcg(mf)

    mf.write_input()
    success, _ = mf.run_model(silent=True)
    if not success:
        raise RuntimeError("MODFLOW failed.")

    hds = bf.HeadFile(Path(workspace) / f"{model_name}.hds")
    head = hds.get_data(totim=hds.get_times()[-1])[0]
    head[head == -999.99] = np.nan

    x = np.linspace(0, Lx, ncol)
    y = np.linspace(0, Ly, nrow)
    X, Y = np.meshgrid(y, x)

    return X, Y, np.flip(head, axis=0)


# -------------------------- #
# 2. Optimization Thread
# -------------------------- #
class PSOThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(float, float, list, object, list)

    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def run(self):
        np.random.seed(self.seed)

        workspace = "D:/Courses Taught/Groundwater Hydrology/GWH Tutorial/Flopy/"
        model_name = "tutorial01_mf"
        true_hk = 3.33
        X, Y, head_true = problem1(hk=true_hk, workspace=workspace, model_name=model_name)

        n_obs = 10
        obs_locs = []
        while len(obs_locs) < n_obs:
            row, col = np.random.randint(0, 20), np.random.randint(0, 20)
            if not np.isnan(head_true[row, col]) and (row, col) not in obs_locs:
                obs_locs.append((row, col))

        obs_heads = [head_true[row, col] for row, col in obs_locs]

        def objective(hk):
            _, _, head_sim = problem1(hk[0], workspace=workspace, model_name=model_name)
            sim_heads = [head_sim[row, col] for row, col in obs_locs]
            return np.sqrt(np.nanmean((np.array(sim_heads) - np.array(obs_heads)) ** 2))

        # Particle Swarm Optimization
        num_particles, max_iter = 20, 10
        lb, ub = 0.1, 10.0
        positions = np.random.uniform(lb, ub, size=(num_particles, 1))
        velocities = np.zeros_like(positions)
        p_best_pos = positions.copy()
        p_best_val = np.array([objective(pos) for pos in positions])

        g_best_index = np.argmin(p_best_val)
        g_best_pos = p_best_pos[g_best_index].copy()
        g_best_val = p_best_val[g_best_index]
        rmse_history = [g_best_val]

        for i in range(max_iter):
            for j in range(num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[j] = (
                    0.5 * velocities[j]
                    + 1.5 * r1 * (p_best_pos[j] - positions[j])
                    + 1.5 * r2 * (g_best_pos - positions[j])
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
            self.progress_signal.emit(int((i + 1) / max_iter * 100))

        _, _, head_opt = problem1(hk=g_best_pos[0], workspace=workspace, model_name=model_name)
        simulated_heads = [head_opt[row, col] for row, col in obs_locs]
        self.finished_signal.emit(
            g_best_pos[0], g_best_val, rmse_history,
            (X, Y, head_true, obs_locs),
            [obs_heads, simulated_heads]
        )


# --------------------------- #
# 3. GUI Application
# --------------------------- #
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Inverse Groundwater Modeling - PSO")
        self.setGeometry(200, 200, 400, 200)

        self.seed_input = QLineEdit("40")
        self.seed_input.setPlaceholderText("Enter random seed...")

        self.progress = QProgressBar()
        self.progress.setValue(0)

        self.start_btn = QPushButton("Run PSO")
        self.start_btn.clicked.connect(self.run_pso)

        self.info_label = QLabel("Enter seed and click 'Run PSO'.")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Random Seed:"))
        layout.addWidget(self.seed_input)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.progress)
        layout.addWidget(self.info_label)

        self.canvas_layout = QVBoxLayout()
        layout.addLayout(self.canvas_layout)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def run_pso(self):
        try:
            seed = int(self.seed_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Seed must be an integer.")
            return

        self.thread = PSOThread(seed)
        self.thread.progress_signal.connect(self.progress.setValue)
        self.thread.finished_signal.connect(self.plot_results)
        self.thread.start()
        self.start_btn.setDisabled(True)

    def plot_results(self, hk, rmse, rmse_history, mesh_data, head_data):
        X, Y, head_true, obs_locs = mesh_data
        obs_heads, sim_heads = head_data

        for i in reversed(range(self.canvas_layout.count())):
            widget = self.canvas_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        def add_plot(fig):
            canvas = FigureCanvas(fig)
            self.canvas_layout.addWidget(canvas)

        # 1. Head Distribution
        fig1, ax1 = plt.subplots()
        cp = ax1.contourf(X, Y, head_true, levels=20, cmap="viridis")
        x_coords = [col * 10 + 5 for row, col in obs_locs]
        y_coords = [row * 10 + 5 for row, col in obs_locs]
        ax1.scatter(x_coords, y_coords, c='red', edgecolor='black', label='Obs Points')
        fig1.colorbar(cp, ax=ax1, label="Head (m)")
        ax1.set_title("Head Distribution with Obs Points")
        add_plot(fig1)

        # 2. RMSE vs Iteration
        fig2, ax2 = plt.subplots()
        ax2.plot(rmse_history, marker='o')
        ax2.set_title("RMSE vs Iteration")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("RMSE")
        add_plot(fig2)

        # 3. Bar Chart
        fig3, ax3 = plt.subplots()
        indices = np.arange(len(obs_heads))
        bar_width = 0.35
        ax3.bar(indices, obs_heads, bar_width, label='Observed', color='green')
        ax3.bar(indices + bar_width, sim_heads, bar_width, label='Simulated', color='orange')
        ax3.set_title("Observed vs Simulated Heads")
        ax3.set_xticks(indices + bar_width / 2)
        ax3.set_xticklabels([f"P{i+1}" for i in range(len(obs_heads))])
        ax3.legend()
        add_plot(fig3)

        self.info_label.setText(f"✅ Optimal hk: {hk:.4f} | RMSE: {rmse:.4f}")
        self.start_btn.setDisabled(False)


# ----------------------- #
# Run the App
# ----------------------- #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
