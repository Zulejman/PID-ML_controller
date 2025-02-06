import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.consys import CONSYS
import numpy as np
import os

def main():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_PATH = os.path.join(BASE_DIR, "code/config")

    plant_config = "/pressure_plant_cfg.json"
    control_config = "/control_cfg.json"
    
    plant_config = CONFIG_PATH + plant_config
    control_config = CONFIG_PATH + control_config

    consys = CONSYS(plant_config, control_config)
    consys.run_simulation()
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, consys.epochs + 1), consys.mse_history, marker='o', linestyle='-')
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Controller Training Progress (MSE)")
    plt.grid(True)
    
    param_history = np.array(consys.param_history)  
    
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

if __name__ == "__main__":
    main()
