import numpy as np
import matplotlib.pyplot as plt

# --- Physical constants ---
R = 2.912       # W·yr/m^2/K
Q = 342.0       # W/m^2
alpha = 0.30
sigma = 5.67e-8 # W/m^2/K^4
epsilon = 0.61  # calibrated emissivity

# --- Derived coefficients ---
heat_input = Q * (1 - alpha) / R
radiative_loss_coeff = epsilon * sigma / R

# --- ODE right-hand side ---
def dT_dt(t, T):
    """Rate of change of global mean temperature (K/year)."""
    return heat_input - radiative_loss_coeff * T**4

# --- Euler’s method ---
def euler_method(T0, t0, t_end, steps_per_year=10, log_file="euler_steps.txt"):
    h = 1.0 / steps_per_year          # fraction of a year covered per step
    N = int((t_end - t0) * steps_per_year)  # total number of steps
    
    t_values = [t0] # time points
    T_values = [T0] # temperature points
    
    with open(log_file, "w") as f: # open log file to document steps and calculations
        f.write(f"Euler's Method with step size h={h}\n\n")
        t, T = t0, T0
        for i in range(N): # iterate through each step
            slope = dT_dt(t, T)
            T_next = T + h * slope
            t_next = t + h
            
            # Write detailed step explanation
            f.write(f"Step {i+1}:\n")
            f.write(f"  Start at (t={t:.2f}, T={T:.4f})\n") # Previous time and temperature point
            f.write(f"  dT/dt = {slope:.6f}\n") # Slope at current point
            f.write(f"  Formula: t_next = t + h = {t:.2f} + {h:.2f} = {t_next:.2f}\n") # Time increment
            f.write(f"           T_next = T + h * dT/dt = {T:.4f} + {h:.2f}*{slope:.6f} = {T_next:.4f}\n\n") # Temp increment
            
            # Store new values
            t_values.append(t_next)
            T_values.append(T_next)
            
            t, T = t_next, T_next # Update current values for next iteration
    
    return np.array(t_values), np.array(T_values)

# --- Run Euler’s method ---
t0, T0 = 0.0, 288.4   # Initial conditions: Year, Temperature (K)
t_end = 5             # Run simulation for 5 years
t_values, T_values = euler_method(T0, t0, t_end, steps_per_year=12) # Euler approximation execution

# --- Plot result ---
plt.figure(figsize=(9,5))
plt.plot(t_values, T_values, "x-", color="#D95F02", label="Euler's Method (12 steps/yr)") # Graph and style the datapoints
plt.xlabel("Time (years elapsed since 2015)", fontsize=12) # X-axis label
plt.ylabel("Temperature (K)", fontsize=12) # Y-axis label
plt.title("Euler Approximation of Global Average Temperature Model", fontsize=14, weight="bold") # Title
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()