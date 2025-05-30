import sys
import re
import sympy as sp
from sympy import symbols, sympify, exp, log
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel, \
    QGraphicsDropShadowEffect
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class SolverApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DE Checker")
        self.setFixedSize(600, 600)
        self.layout = QVBoxLayout()
        self.title_label = QLabel("Differential Equation Checker", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
                    background-color: #D5CAEB;
                    font-size: 20px; 
                    font-weight: bold; 
                    color: #000000; 
                    padding: 10px;
                """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setXOffset(3)
        shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 160))
        self.title_label.setGraphicsEffect(shadow)
        self.layout.addWidget(self.title_label)
        self.eq_input = QLineEdit(self)
        self.eq_input.setPlaceholderText("Enter (M)dx + (N)dy = 0")
        self.layout.addWidget(self.eq_input)
        self.eq_input.setStyleSheet(""" 
            background-color: #D3D3D3; 
            color: #333333; 
            padding: 5px; 
            border: 2px solid #888888; 
            border-radius: 5px;
        """)
        self.check_btn = QPushButton("Check", self)
        self.check_btn.clicked.connect(self.check_eq)
        self.layout.addWidget(self.check_btn)
        self.check_btn.setStyleSheet("""
            background-color: #D5CAEB; 
            color: black; 
            padding: 10px; 
            font-size: 20px; 
            border-radius: 5px;
        """)
        self.result_label = QLabel("", self)
        self.layout.addWidget(self.result_label)
        self.result_label.setStyleSheet("""
            color: #333333; 
            font-size: 14px;
        """)
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)
        self.plot_empty_cartesian()
        self.set_style()

    def set_style(self):
        palette = QPalette()
        palette.setColor(QPalette.Background, QColor("#D5CAEB"))
        self.setPalette(palette)
        self.setStyleSheet("""
            border: 2px solid #000000;
        """)

    def plot_empty_cartesian(self):
        self.ax.clear()
        self.fig.patch.set_facecolor('#D5CAEB')
        self.ax.set_facecolor('#F0F0F0')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Solution Curve of the Differential Equation')
        self.ax.grid(True, color='darkblue')
        self.canvas.draw()

    def plot_solution(self, solution):
        self.ax.clear()
        self.fig.patch.set_facecolor('#D5CAEB')
        self.ax.set_facecolor('#F0F0F0')
        self.ax.plot(solution['x_vals'], solution['y_vals'], label="Equation", color='red')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Solution Curve of the Differential Equation')
        self.ax.legend()
        self.ax.grid(True, color='blue')
        self.canvas.draw()

    def check_eq(self):
        def contains_exp_or_log(M, N):
            if any(isinstance(term, sp.exp) for term in M.as_ordered_terms()):
                return True
            if any(isinstance(term, sp.exp) for term in N.as_ordered_terms()):
                return True
            if any(isinstance(term, sp.log) for term in M.as_ordered_terms()):
                return True
            if any(isinstance(term, sp.log) for term in N.as_ordered_terms()):
                return True
            return False

        eq = self.eq_input.text()
        try:
            M, N = parse_eq(eq)
            if contains_exp_or_log(M, N):
                self.result_label.setText(self.result_label.text() + "\nEquation contains exp or log, not plotting.")
                self.plot_empty_cartesian()
                return

            exact, exact_msg = check_exact(M, N)
            self.result_label.setText(exact_msg)
            homo, homo_msg = check_homo(M, N)
            self.result_label.setText(self.result_label.text() + "\n" + homo_msg)

            if exact and not contains_exp_or_log(M, N):
                solution = solve_de(M, N)
                self.plot_solution(solution)
            else:
                self.result_label.setText(self.result_label.text() + "\nCannot plot, equation is not exact.")
                self.plot_empty_cartesian()

        except ValueError as e:
            self.result_label.setText(self.result_label.text() + "\nCannot plot, equation contains exp or log")
            self.plot_empty_cartesian()
            pass
        except Exception as e:
            self.result_label.setText(self.result_label.text() + "\nCannot plot, equation contains exp or log")
            self.plot_empty_cartesian()
            pass


def parse_eq(eq):
    eq = eq.replace("ln(", "log(")
    eq = eq.replace(" ", "")
    eq = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', eq)
    eq = eq.replace("^", "**")
    match = re.match(r'\((.*?)\)dx\+\((.*?)\)dy=0', eq)
    if match:
        M = match.group(1)
        N = match.group(2)

        if is_log_or_exp(M) or is_log_or_exp(N):
            return None, None

        M = sp.sympify(M)
        N = sp.sympify(N)
        return M, N
    else:
        raise ValueError("Invalid equation format.")


def check_exact(M, N):
    x, y = symbols('x y')
    M = sympify(M)
    N = sympify(N)
    dM_dy = M.diff(y)
    dN_dx = N.diff(x)
    if dM_dy == dN_dx:
        return True, f"Exact: dM/dy = {dM_dy} and dN/dx = {dN_dx}"
    else:
        return False, f"Not exact: dM/dy = {dM_dy} and dN/dx = {dN_dx}"


def check_homo(M, N):
    x, y = symbols('x y')
    M = sympify(M)
    N = sympify(N)
    if any(is_log_or_exp(term) for term in [M, N]):
        return False, "Not homogeneous due to log/exp terms"
    try:
        deg_M = total_degree(M, x, y)
        deg_N = total_degree(N, x, y)
    except:
        return False, "Not homogeneous"
    if deg_M == deg_N:
        return True, f"Homogeneous with degree {deg_M}"
    else:
        return False, "Not homogeneous"


def total_degree(expr, x, y):
    total_deg = 0
    for term in expr.as_ordered_terms():
        deg_x = 0
        deg_y = 0
        if term.has(x):
            deg_x = term.as_expr().as_poly(x).degree()
        if term.has(y):
            deg_y = term.as_expr().as_poly(y).degree()
        total_deg = max(total_deg, deg_x + deg_y)
    return total_deg


def is_log_or_exp(term):
    return isinstance(term, log) or isinstance(term, exp)


def solve_de(M, N):
    x, y = symbols('x y')
    M_func = sp.lambdify((x, y), M, "numpy")
    N_func = sp.lambdify((x, y), N, "numpy")

    def model(y, x):
        dydx = -M_func(x, y) / N_func(x, y)
        return dydx

    x_vals = np.linspace(0, 10, 100)
    y0 = 1
    y_vals = odeint(model, y0, x_vals)
    return {'y_vals': y_vals[:, 0], 'x_vals': x_vals}


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SolverApp()
    window.show()
    sys.exit(app.exec_())
