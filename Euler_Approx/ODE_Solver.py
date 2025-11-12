import jax.numpy as jnp
import matplotlib.pyplot as plt
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

# --- ODE definition ---
def climate_ode(t, temperature, args):
    """Simple energy balance model: dT/dt = incoming - outgoing."""
    return heat_input - radiative_loss_coeff * temperature**4

term = ODETerm(climate_ode) # Wrap ODE function in a term object (Diffrax requirement)
solver = Dopri5()           # Choose Dormand–Prince 5th-order Runge–Kutta solver (similar to RK45 in SciPy)

# --- Initial condition & solver setup ---
T_init = jnp.array(288.4)     # Initial condition: global mean temperature (K)
t0, t1 = 0, 5                 # Time span: simulate from year 0 to year 5
dt0 = 0.1                     # Initial guess for step size (solver adapts from here)
save_points = jnp.linspace(t0, t1, 500) # Store 500 equally spaced points between t0 and t1

# --- Run Diffrax solver ---
solution = diffeqsolve(
    term, solver, t0=t0, t1=t1, dt0=dt0, y0=T_init, saveat=SaveAt(ts=save_points)
)

# --- Plot result ---
plt.figure(figsize=(9, 5))
plt.plot(solution.ts, solution.ys, color="#55C297", linewidth=2.5, label="Diffrax ODE Solver") # Continuous solution curve

plt.xlabel("Time (years elapsed since 2015)", fontsize=12) # X-axis label
plt.ylabel("Temperature (K)", fontsize=12)                 # Y-axis label
plt.title("ODE Solver for Global Average Temperature Model", fontsize=14, weight="bold") # Title
plt.legend(frameon=False) # Add legend with clean styling
plt.grid(alpha=0.3)       # Light grid for readability
plt.tight_layout()
plt.show()