import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QComboBox, QFrame, QFormLayout
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import your model logic
from patmodels import pat_initalcondn, pat_model, pump_treat_objective, pso_pat


class GroundwaterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pump and Treat Tool")
        self.resize(1200, 700)

        self.head = None
        self.conc = None
        self.X = None
        self.Y = None

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Left panel - controls
        controls = QVBoxLayout()

        # Load Model & Simulate button (placed above the dropdown)
 

        # File inputs
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
        
        self.load_button = QPushButton("Load Model & Simulate")
        self.load_button.clicked.connect(self.load_and_simulate)
        file_layout.addWidget(self.load_button)

        controls.addLayout(file_layout)

        # Dropdown for optimization selection
        self.optimization_dropdown = QComboBox()
        self.optimization_dropdown.addItem("Select Optimization")
        self.optimization_dropdown.addItem("Manual Optimization")
        self.optimization_dropdown.addItem("Automatic Optimization")
        controls.addWidget(self.optimization_dropdown)

        # Proceed button
        self.proceed_button = QPushButton("Proceed")
        self.proceed_button.clicked.connect(self.proceed_optimization)
        controls.addWidget(self.proceed_button)

        # Spacer
        controls.addStretch()

        # Right panel - Matplotlib plots
        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)

        # Add both panels to the main layout
        main_layout.addLayout(controls, 1)
        main_layout.addWidget(self.canvas, 3)

        self.setLayout(main_layout)

        # Bottom frame for dynamic input fields after optimization selection
        self.dynamic_frame = QFrame(self)
        self.dynamic_layout = QFormLayout(self.dynamic_frame)
        self.dynamic_frame.setLayout(self.dynamic_layout)
        self.dynamic_frame.setVisible(False)  # Initially hidden
        controls.addWidget(self.dynamic_frame)

    def create_browse_button(self, target_line_edit):
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.select_file(target_line_edit))
        return btn

    def select_file(self, target_line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_path:
            target_line_edit.setText(file_path)

    def load_and_simulate(self):
        ibound_path = self.ibound_input.text()
        icbund_path = self.icbund_input.text()
        strt_path = self.strt_input.text()

        self.head, self.conc, self.X, self.Y = pat_initalcondn(ibound_path, icbund_path, strt_path)
        self.update_simulation_plot()

    def proceed_optimization(self):
        selected_optimization = self.optimization_dropdown.currentText()

        if selected_optimization == "Select Optimization":
            return  # Do nothing

        # Hide previous dynamic inputs if any
        self.clear_dynamic_inputs()

        if selected_optimization == "Manual Optimization":
            self.show_manual_optimization_fields()
        elif selected_optimization == "Automatic Optimization":
            self.show_automatic_optimization_fields()

    def show_manual_optimization_fields(self):
        # Show inputs for Manual Optimization: locations and rates of pumps
        self.dynamic_frame.setVisible(True)

        self.pump_location_input = QLineEdit()
        self.pump_rate_input = QLineEdit()

        self.dynamic_layout.addRow(QLabel("Enter Pump Locations (x, y):"), self.pump_location_input)
        self.dynamic_layout.addRow(QLabel("Enter Pump Rates:"), self.pump_rate_input)

        # Add Optimize button
        self.optimize_button = QPushButton("Optimize")
        self.optimize_button.clicked.connect(self.manual_opt)
        self.dynamic_layout.addWidget(self.optimize_button)

    def show_automatic_optimization_fields(self):
        # Show inputs for Automatic Optimization: locations and bounds for pumps
        self.dynamic_frame.setVisible(True)

        self.pump_location_input = QLineEdit()
        self.lower_bound_input = QLineEdit()
        self.upper_bound_input = QLineEdit()

        self.dynamic_layout.addRow(QLabel("Enter Pump Locations (x, y):"), self.pump_location_input)
        self.dynamic_layout.addRow(QLabel("Enter Lower Bound for Pump Rates:"), self.lower_bound_input)
        self.dynamic_layout.addRow(QLabel("Enter Upper Bound for Pump Rates:"), self.upper_bound_input)

        # Add Optimize button
        self.optimize_button = QPushButton("Optimize")
        self.optimize_button.clicked.connect(self.auto_opt)
        self.dynamic_layout.addWidget(self.optimize_button)

    def clear_dynamic_inputs(self):
        # Clear dynamic fields
        for i in reversed(range(self.dynamic_layout.count())):
            widget = self.dynamic_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

    def manual_opt(self):
        # Get the pump locations and rates for manual optimization
        locations = self.pump_location_input.text()
        rates = self.pump_rate_input.text()

        # Assuming `pat_model` accepts the pump rates as a list of values
        Q1, Q2, Q3 = map(float, rates.split(','))  # Modify as per actual input format
        conc_ = pat_model(Q1, Q2, Q3)
        self.update_concentration_plot(conc_)

    def auto_opt(self):
        # Get the pump locations and bounds for automatic optimization
        locations = self.pump_location_input.text()
        
        lb1,lb2,lb3 = map(float, self.lower_bound_input.text().split(','))
        ub1,ub2,ub3= map(float, self.upper_bound_input.text().split(','))
        
        best_pumping, final_cost, history = pso_pat(pump_treat_objective, [lb1,lb2,lb3], [ub1,ub2,ub3], max_iter=10)

        conc_ = pat_model(*best_pumping)
        self.update_concentration_plot(conc_)
        self.plot_cost_history(history)

    def update_simulation_plot(self):
        if self.head is None or self.conc is None:
            return

        head = np.where(self.head == -999.99, np.nan, self.head)
        conc = np.where(self.conc == 1e+30, np.nan, self.conc)

        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)

        c1 = ax1.contourf(self.X, self.Y, head, cmap="viridis")
        ax1.set_title("Hydraulic Head")
        self.figure.colorbar(c1, ax=ax1)

        c2 = ax2.contourf(self.X, self.Y, conc, cmap="inferno")
        ax2.set_title("Contaminant Concentration")
        self.figure.colorbar(c2, ax=ax2)

        self.canvas.draw()

    def update_concentration_plot(self, conc_):
     
        conc_ = np.where(conc_ == 1e+30, np.nan, conc_)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        cs = ax.contourf(self.X, self.Y, conc_, cmap='jet')
        ax.set_title("Contaminant Plume after Remediation")
        self.figure.colorbar(cs, ax=ax)
        self.canvas.draw()

    def plot_cost_history(self, history):
        fig, ax = plt.subplots()
        ax.plot(history, marker='o')
        ax.set_title("Cost vs Iteration (PSO)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GroundwaterGUI()
    gui.show()
    sys.exit(app.exec_())
