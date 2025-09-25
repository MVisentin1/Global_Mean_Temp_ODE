import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt

# --- Physical constants ---
R = 2.912       # W·yr/m^2/K
Q = 342.0       # W/m^2
alpha = 0.30
sigma = 5.67e-8 # W/m^2/K^4
epsilon = 0.61  # calibrated emissivity

# --- Derived coefficients ---
heat_input = Q * (1 - alpha) / R                # Incoming solar energy, adjusted by albedo
radiative_loss_coeff = epsilon * sigma / R      # Outgoing energy proportional to T^4

# --- ODE right-hand side ---
def dT_dt(t, T):
    """Rate of change of global mean temperature (K/year)."""
    return heat_input - radiative_loss_coeff * T**4

# --- Diffrax ODE definition ---
def climate_ode(t, temperature, args):
    """Diffrax requires (t, y, args) format."""
    return dT_dt(t, temperature)

# --- High-accuracy solver (Diffrax) ---
term = ODETerm(climate_ode)  # Wrap ODE in a Diffrax term
solver = Dopri5()            

T_init = jnp.array(288.4)    # Initial condition: observed mean global temperature in 2015 (K)
t0, t1 = 0, 5                # Simulate from 2015 (t=0) through 2020 (t=5)
dt0 = 0.1                    # Initial step size guess
save_points = jnp.linspace(t0, t1, 500) # Store 500 points for smooth plotting

# Run Diffrax solver
solution = diffeqsolve(
    term, solver, t0=t0, t1=t1, dt0=dt0, y0=T_init, saveat=SaveAt(ts=save_points)
)

# --- Euler’s method implementation ---
def euler_method(T0, t0, t_end, steps_per_year=12):
    h = 1.0 / steps_per_year              # Fraction of a year per step (monthly steps)
    N = int((t_end - t0) * steps_per_year) # Total number of steps
    
    t_values = [t0] # Store time points
    T_values = [T0] # Store temperature points
    
    t, T = t0, T0
    for _ in range(N): # Iterate through each step
        slope = dT_dt(t, T)                 # Evaluate slope at current point
        T_next = T + h * slope              # Update temperature using Euler’s method
        t_next = t + h                      # Advance time
        t_values.append(t_next)
        T_values.append(T_next)
        t, T = t_next, T_next               # Prepare for next iteration
    
    return np.array(t_values), np.array(T_values)

# Run Euler’s method (monthly steps: 12 per year)
t_values, T_values = euler_method(288.4, 0.0, 5, steps_per_year=12)

# --- NASA observational datapoints ---
baseline_K = 287.56 # 1950's - 1990's approx. mean global temperature in Kelvin
nasa_data = {
    2015: baseline_K + 0.83,
    2016: baseline_K + 0.88,
    2017: baseline_K + 0.91,
    2018: baseline_K + 0.93,
    2019: baseline_K + 0.94,
    2020: baseline_K + 0.97,
}
# Convert years to relative t (2015 → 0, 2016 → 1, etc.)
nasa_t = np.array([year - 2015 for year in nasa_data.keys()])
nasa_T = np.array(list(nasa_data.values()))

# --- Plot all results together ---
plt.figure(figsize=(9,5))

# Plot Diffrax solver (reference solution)
plt.plot(solution.ts, solution.ys, color="#55C297", linewidth=2.5, label="Diffrax ODE Solver")

# Plot Euler approximation
plt.plot(t_values, T_values, color="#D95F02", label="Euler’s Method (12 steps/yr)")

# Plot NASA observational datapoints
plt.plot(nasa_t, nasa_T, 'x-', color="#5390AC", linewidth=2.0, markersize=6, label="NASA GISS Observations (2015–2020)")

# Axis labels and title
plt.xlabel("Time (years elapsed since 2015)", fontsize=12) # X-axis label
plt.ylabel("Temperature (K)", fontsize=12)                 # Y-axis label
plt.title("ODE Solver vs. Euler Approximation vs. NASA Data", fontsize=14, weight="bold") # Title

# Plot styling
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()