import matplotlib
from consys import CONSYS  # Import the CONSYS class
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Initialize CONSYS with config files
plant = "plant_cfg.json"
control = "control_cfg.json"

consys = CONSYS(plant, control)

# Run the simulation
consys.run_simulation()

plt.figure(figsize=(8,5))
plt.plot(range(1, consys.epochs + 1), consys.mse_history, marker='o', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Controller Training Progress")
plt.grid(True)
plt.savefig("mse_plot.png")
