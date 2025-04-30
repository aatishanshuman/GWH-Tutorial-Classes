import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QComboBox, QFrame, QFormLayout,
    QTextEdit
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from patmodels import *


class GroundwaterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pump and Treat Tool")
        self.resize(1300, 800)

        self.head = self.conc = self.X = self.Y = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()  # Top-level layout

        # Top content: controls and plots
        top_layout = QHBoxLayout()

        # Left panel - controls
        controls = QVBoxLayout()

        file_layout = QGridLayout()
        self.ibound_input = QLineEdit()
        self.icbund_input = QLineEdit()
        self.strt_input = QLineEdit()

        file_layout.addWidget(QLabel("ibound path:"), 0, 0)
        file_layout.addWidget(self.ibound_input, 0, 1)
        file_layout.addWidget(self.create_browse_button(self.ibound_input), 0, 2)

        file_layout.addWidget(QLabel("icbund path:"), 1, 0)
        file_layout.addWidget(self.icbund_input, 1, 1)
        file_layout.addWidget(self.create_browse_button(self.icbund_input), 1, 2)

        file_layout.addWidget(QLabel("strt path:"), 2, 0)
        file_layout.addWidget(self.strt_input, 2, 1)
        file_layout.addWidget(self.create_browse_button(self.strt_input), 2, 2)

        self.load_button = QPushButton("Simulate Pollution")
        self.load_button.clicked.connect(self.load_and_simulate)
        file_layout.addWidget(self.load_button, 3, 0, 1, 3)

        controls.addLayout(file_layout)

        # Optimization dropdown and proceed
        self.optimization_dropdown = QComboBox()
        self.optimization_dropdown.addItems(["Select Optimization", "Manual Optimization", "Automatic Optimization"])
        controls.addWidget(self.optimization_dropdown)

        self.proceed_button = QPushButton("Proceed")
        self.proceed_button.clicked.connect(self.proceed_optimization)
        controls.addWidget(self.proceed_button)

        # Dynamic form for optimization inputs
        self.dynamic_frame = QFrame()
        self.dynamic_layout = QFormLayout(self.dynamic_frame)
        self.dynamic_frame.setVisible(False)
        controls.addWidget(self.dynamic_frame)

        controls.addStretch()

        # Right panel - Main plots
        plot_panel = QVBoxLayout()
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        plot_panel.addWidget(self.canvas)



        top_layout.addLayout(controls, 1)
        top_layout.addLayout(plot_panel, 3)

        # Bottom - console output
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("background-color: #111; color: #0f0; font-family: Consolas;")

        main_layout.addLayout(top_layout)
        main_layout.addWidget(QLabel("Console Output:"))
        main_layout.addWidget(self.console_output)

        self.setLayout(main_layout)

    def create_browse_button(self, target_line_edit):
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.select_file(target_line_edit))
        return btn

    def select_file(self, target_line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_path:
            target_line_edit.setText(file_path)

    def load_and_simulate(self):
        self.ibound_path = self.ibound_input.text()
        self.icbund_path = self.icbund_input.text()
        self.strt_path = self.strt_input.text()

        self.head, self.conc, self.X, self.Y = pat_initalcondn(self.ibound_path, self.icbund_path, self.strt_path)
        self.update_simulation_plot()
        self.log("Model loaded and initial simulation completed.")

    def proceed_optimization(self):
        self.clear_dynamic_inputs()
        self.dynamic_frame.setVisible(True)

        opt_type = self.optimization_dropdown.currentText()

        if opt_type == "Manual Optimization":
            self.pump_rate_input = QLineEdit()
            self.dynamic_layout.addRow(QLabel("Enter Pump Rates (comma separated):"), self.pump_rate_input)
            self.optimize_button = QPushButton("Optimize")
            self.optimize_button.clicked.connect(self.manual_opt)

        elif opt_type == "Automatic Optimization":
            self.lower_bound_input = QLineEdit()
            self.upper_bound_input = QLineEdit()
            self.dynamic_layout.addRow(QLabel("Lower Bounds (comma separated):"), self.lower_bound_input)
            self.dynamic_layout.addRow(QLabel("Upper Bounds (comma separated):"), self.upper_bound_input)
            self.optimize_button = QPushButton("Optimize")
            self.optimize_button.clicked.connect(self.auto_opt)

        self.dynamic_layout.addWidget(self.optimize_button)

    def manual_opt(self):
        try:
            rates = list(map(float, self.pump_rate_input.text().split(',')))
            conc_ = pat_model(*rates,self.ibound_path, self.icbund_path, self.strt_path)
            self.update_concentration_plot(conc_,mode='manual')
            
            cost=pump_treat_objective(rates, self.ibound_path, self.icbund_path, self.strt_path)
            
            self.log(f"Manual optimization completed with rates: {rates} Total cost:{cost}")
            
            
        except Exception as e:
            self.log(f"[ERROR] Manual Optimization Failed: {e}")

    def auto_opt(self):
        try:
            lb = list(map(float, self.lower_bound_input.text().split(',')))
            ub = list(map(float, self.upper_bound_input.text().split(',')))
            self.log("Automatic optimization started. Please wait till completion.")

            best_pumping, final_cost, history = pso_pat(
                lambda Q: pump_treat_objective(Q, self.ibound_path, self.icbund_path, self.strt_path),
                lb, ub
            )
            conc_ = pat_model(*best_pumping,self.ibound_path, self.icbund_path, self.strt_path)
            
            

            self.update_concentration_plot(conc_, mode='automatic',history=history)
            self.log(f"Automatic Optimization Completed.\nBest Pumping Rates: {best_pumping}\nFinal Cost: {final_cost:.2f}")
        except Exception as e:
            self.log(f"[ERROR] Automatic Optimization Failed: {e}")

    def clear_dynamic_inputs(self):
        for i in reversed(range(self.dynamic_layout.count())):
            widget = self.dynamic_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

    def update_simulation_plot(self):
        if self.head is None or self.conc is None:
              return
      
        head = np.where(self.head == -999.99, np.nan, self.head)
        conc = np.where(self.conc == 1e+30, np.nan, self.conc)
        
        self.figure.clear()
        
        # Create subplots with shared X and Y labels
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)
        
        # --- Hydraulic Head Plot ---
        levels_head = np.linspace(np.nanmin(head), np.nanmax(head), 15)
        cf1 = ax1.contourf(self.X, self.Y, head, levels=levels_head, cmap="viridis", alpha=0.8)
        c1 = ax1.contour(self.X, self.Y, head, levels=levels_head, colors='k', linewidths=0.8)
        ax1.clabel(c1, inline=True, fontsize=10, fmt="%.1f", inline_spacing=5)
        
        # Add colorbar and label it
        cbar1 = self.figure.colorbar(cf1, ax=ax1, pad=0.01)
        cbar1.set_label("Head (m)", fontsize=10)
        
        # Add custom legend for Hydraulic Head
        ax1.plot([], [], color='black', label="Contour Lines (Head)", linewidth=0.8)  # Empty plot for legend
        ax1.plot([], [], color='yellow', label="Contours of Hydraulic Head", linewidth=8)  # Dummy for color
    
        # Titles, labels, and aspect for head plot
        ax1.set_title("Hydraulic Head", fontsize=14, weight='bold')
        ax1.set_xlabel("X (m)", fontsize=12)
        ax1.set_ylabel("Y (m)", fontsize=12)
        ax1.set_aspect("equal", adjustable='box')
        
        # --- Contaminant Concentration Plot ---
        levels_conc = np.linspace(np.nanmin(conc), np.nanmax(conc), 15)
        cf2 = ax2.contourf(self.X, self.Y, conc, levels=levels_conc, cmap="inferno", alpha=0.8)
        c2 = ax2.contour(self.X, self.Y, conc, levels=levels_conc, colors='k', linewidths=0.8)
        ax2.clabel(c2, inline=True, fontsize=10, fmt="%.1f", inline_spacing=5)
        
        # Add colorbar and label it
        cbar2 = self.figure.colorbar(cf2, ax=ax2, pad=0.01)
        cbar2.set_label("Concentration (ppm)", fontsize=10)
        
        # Add custom legend for Contaminant Concentration
        ax2.plot([], [], color='black', label="Contour Lines (Concentration)", linewidth=0.8)  # Empty plot for legend
        ax2.plot([], [], color='red', label="Contours of Concentration", linewidth=8)  # Dummy for color
    
        # Titles, labels, and aspect for concentration plot
        ax2.set_title("Contaminant Concentration", fontsize=14, weight='bold')
        ax2.set_xlabel("X (m)", fontsize=12)
        ax2.set_ylabel("Y (m)", fontsize=12)
        ax2.set_aspect("equal", adjustable='box')

        
        # Tight layout for the figure to avoid overlapping
        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()


    def update_concentration_plot(self, conc_, mode=None, history=None):
        conc_ = np.where(conc_ == 1e+30, np.nan, conc_)
        self.figure.clear()
        
    
        
        if mode == 'manual':
            ax = self.figure.add_subplot(111)
            levels = np.linspace(np.nanmin(conc_), np.nanmax(conc_), 10)
    
            cf = ax.contourf(self.X, self.Y, conc_, levels=levels, cmap='jet')
            c = ax.contour(self.X, self.Y, conc_, levels=levels, colors='k', linewidths=0.5)
            ax.clabel(c, inline=True, fontsize=8, fmt="%.1f")
    
            ax.set_title("Contaminant Plume after Remediation", fontsize=12)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            self.figure.colorbar(cf, ax=ax, label="Concentration (ppm)")
    
            # Plot Wells (Wells are marked with 'r*' - red stars)
            for well in wells_xy:
                ax.scatter(well[1], well[2], color='red', marker='*', label=well[0], s=100)
            
            # Plot Monitoring Wells (Monitoring Wells are marked with 'bs' - blue squares)
            for mw in monitoring_wells_xy:
                ax.scatter(mw[1], mw[2], color='lime', marker='s', label=mw[0], s=80)
            
            # Plot Injection Well (Injection Well is marked with 'g^' - green triangle)
            for iw in iwell_xy:
                ax.scatter(iw[1], iw[2], color='green', marker='^', label=iw[0], s=100)
    
            # Add Legend
            ax.legend(loc='upper right', fontsize=10)
    
        elif mode == 'automatic':
            # Plot concentration
            ax1 = self.figure.add_subplot(211)
            levels = np.linspace(np.nanmin(conc_), np.nanmax(conc_), 10)
    
            cf = ax1.contourf(self.X, self.Y, conc_, levels=levels, cmap='jet')
            c = ax1.contour(self.X, self.Y, conc_, levels=levels, colors='k', linewidths=0.5)
            ax1.clabel(c, inline=True, fontsize=8, fmt="%.1f")
    
            ax1.set_title("Contaminant Plume after Remediation", fontsize=12)
            ax1.set_xlabel("X (m)")
            ax1.set_ylabel("Y (m)")
            self.figure.colorbar(cf, ax=ax1, label="Concentration (ppm)")
    
            # Plot Wells (Wells are marked with 'r*' - red stars)
            for well in wells_xy:
                ax1.scatter(well[1], well[2], color='red', marker='*', label=well[0], s=100)
            
            # Plot Monitoring Wells (Monitoring Wells are marked with 'bs' - blue squares)
            for mw in monitoring_wells_xy:
                ax1.scatter(mw[1], mw[2], color='lime', marker='s', label=mw[0], s=80)
            
            # Plot Injection Well (Injection Well is marked with 'g^' - green triangle)
            for iw in iwell_xy:
                ax1.scatter(iw[1], iw[2], color='green', marker='^', label=iw[0], s=100)
    
            # Add Legend
            ax1.legend(loc='upper right', fontsize=10)
    
            # Plot cost vs iteration
            ax2 = self.figure.add_subplot(212)
            ax2.plot(history, marker='o', linestyle='-', color='darkgreen')
            ax2.set_title("Cost vs Iteration (PSO)", fontsize=12)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Cost")
            ax2.grid(True)
    
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()


         

    # def plot_cost_history(self, history):
    #     self.cost_figure.clear()
    #     ax = self.cost_figure.add_subplot(111)

    #     self.cost_canvas.draw()

    def log(self, message):
        self.console_output.append(message)
        print(message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GroundwaterGUI()
    gui.show()
    sys.exit(app.exec_())
