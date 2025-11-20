import numpy as np
import matplotlib.pyplot as plt
from numpy.f2py.auxfuncs import throw_error
from numpy.ma.core import divide

from Euler_Approx.Euler_Approx import euler_method
from Heun_Approx.Heun_Approx_Tool import heun_method

# --- Computes array of relative error of euler's method compared to heun's method
def relative_error(euler_values, heun_values):
    error = []
    for i in range (len(euler_values)):
        error.append(divide(abs(euler_values[i] - heun_values[i]), heun_values[i]))
    return error

# --- Run Heun’s method ---
t0, T0 = 0.0, 288.4   # Initial conditions: Year, Temperature (K)
t_end = 5             # Run simulation for 5 years
t_heun, T_heun = heun_method(T0, t0, t_end, steps_per_year=12)  # Heun approximation execution

# --- Run Euler's method ---
t_euler, T_euler = euler_method(T0, t0, t_end, steps_per_year=12)

# --- Compute Error ---
error = relative_error(T_euler, T_heun)

# --- Plot result ---
plt.figure(figsize=(9,5))
plt.plot(t_heun, error, "x-", color="#1B9E77")
plt.xlabel("Time (years elapsed since 2015)", fontsize=12)
plt.ylabel("Relative Error", fontsize=12)
plt.title("Error Graph (Euler vs. Heun Approximation)", fontsize=14, weight="bold")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("relative_error_comparison.png")
plt.show()

