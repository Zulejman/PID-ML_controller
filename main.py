import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.consys import CONSYS
import numpy as np
import json

def main():

    plant_config = "config/pressure_plant_cfg.json"
    control_config = "config/pid_control_cfg.json"

    consys = CONSYS(plant_config, control_config)
    consys.run_simulation()
    
    with open(control_config) as ccfg:
        fjson = json.load(ccfg)

    if fjson.get("controller") == "PID":

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

    elif fjson.get("controller") == "NN":

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        plt.plot(range(1, consys.epochs + 1), consys.mse_history, marker='o', linestyle='-')
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.title("Controller Training Progress (MSE)")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(range(1, consys.epochs + 1), consys.param_history, marker='o', linestyle='-', label="Param Norm")
        plt.xlabel("Epochs")
        plt.ylabel("Global Parameter Norm")
        plt.title("Evolution of NN Parameter Norm")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("training_plots.png")
        print("Plots saved as training_plots.png.")

    else:
        print("Wrong controler config!")

if __name__ == "__main__":
    main()
