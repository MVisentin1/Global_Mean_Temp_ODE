import numpy as np
import matplotlib.pyplot as plt

# --- Physical constants - to 4 Signifiant Figures ---
R = 2.912       # W·yr/m^2/K
Q = 342.0       # W/m^2
alpha = 0.3000
sigma = 5.670e-8 # W/m^2/K^4
epsilon = 0.6100  # calibrated emissivity

# --- Derived coefficients ---
heat_input = Q * (1 - alpha) / R
radiative_loss_coeff = epsilon * sigma / R

# --- ODE right-hand side ---
def dT_dt(t, T):
    """Rate of change of global mean temperature (K/year)."""
    return heat_input - radiative_loss_coeff * T**4

# --- Heun’s method ---
def heun_method(T0, t0, t_end, steps_per_year=10, log_file="heun_steps.txt"):
    h = 1.0 / steps_per_year          # fraction of a year covered per step
    N = int((t_end - t0) * steps_per_year)  # total number of steps
    
    t_values = [t0]
    T_values = [T0]
    
    with open(log_file, "w") as f:
        f.write(f"Heun's Method with step size h={h}\n\n")
        t, T = t0, T0
        for i in range(N):
            slope_i = dT_dt(t, T)
            T_est = T + h * slope_i
            slope_next = dT_dt(t + h, T_est)
            avg_slope = 0.5 * (slope_i + slope_next)
            T_next = T + h * avg_slope
            t_next = t + h

            # Step-by-step explanation (elementary phrasing)
            f.write(f"Step {i+1}:\n")
            f.write(f"  Point i: (t={t:.2f}, T={T:.4f})\n")
            f.write(f"  Slope at i = {slope_i:.6f}\n")
            f.write(f"  Estimate T at next point = {T:.4f} + {h:.2f}*{slope_i:.6f} = {T_est:.4f}\n")
            f.write(f"  Slope at i+1 (using T_est) = {slope_next:.6f}\n")
            f.write(f"  Average slope = (slope_i + slope_i+1)/2 = {avg_slope:.6f}\n")
            f.write(f"  Point i+1: t = {t:.2f} + {h:.2f} = {t_next:.2f}\n")
            f.write(f"             T = {T:.4f} + {h:.2f}*{avg_slope:.6f} = {T_next:.4f}\n\n")
            
            t_values.append(t_next)
            T_values.append(T_next)
            
            t, T = t_next, T_next
    
    return np.array(t_values), np.array(T_values)

# --- Run Heun’s method ---
t0, T0 = 0.0, 288.4   # Initial conditions: Year, Temperature (K)
t_end = 5             # Run simulation for 5 years
t_values, T_values = heun_method(T0, t0, t_end, steps_per_year=12)  # Heun approximation execution

# --- Plot result ---
plt.figure(figsize=(9,5))
plt.plot(t_values, T_values, "x-", color="#1B9E77", label="Heun's Method (12 steps/yr)")
plt.xlabel("Time (years elapsed since 2015)", fontsize=12)
plt.ylabel("Temperature (K)", fontsize=12)
plt.title("Heun Approximation of Global Average Temperature Model", fontsize=14, weight="bold")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()