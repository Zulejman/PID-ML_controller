import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from consys import CONSYS

def main():
    # Configuration files.
    plant_config = "plant_cfg.json"
    control_config = "control_cfg.json"
    
    # Initialize the system.
    consys = CONSYS(plant_config, control_config)
    
    # Run the simulation/training.
    consys.run_simulation()
    
    # Plot the MSE over epochs.
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, consys.epochs + 1), consys.mse_history, marker='o', linestyle='-')
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Controller Training Progress (MSE)")
    plt.grid(True)
    
    # Convert parameter history to a JAX array (or NumPy array).
    import numpy as np
    param_history = np.array(consys.param_history)  # Shape: (epochs, 3)
    
    # Plot the evolution of the three PID parameters.
    plt.subplot(2, 1, 2)
    plt.plot(range(1, consys.epochs + 1), param_history[:, 0], label="k_p", marker='o')
    plt.plot(range(1, consys.epochs + 1), param_history[:, 1], label="k_i", marker='o')
    plt.plot(range(1, consys.epochs + 1), param_history[:, 2], label="k_d", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Parameter Value")
    plt.title("Evolution of PID Parameters")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_plots.png")
    print("Plots saved as training_plots.png.")

if __name__ == "__main__":
    main()
