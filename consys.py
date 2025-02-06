import jax.numpy as jnp 
import jax
import importlib
import json
import optax

class CONSYS:
    def __init__(self, plant_config, control_config):

        with open(plant_config, "r") as file:
            plant_data = json.load(file)

        with open(control_config, "r") as file:
            control_data = json.load(file)

        plant_module_name = plant_data.get("module", "plant")  
        plant_class_name = plant_data.get("plant_class", "Bathtub")
        plant_module = importlib.import_module(plant_module_name)
        plant_class = getattr(plant_module, plant_class_name)  
        self.plant = plant_class(plant_config)  

        controller_type = control_data.get("controller", "PID")
        self.epochs = control_data.get("epochs", 100)
        self.timesteps = control_data.get("steps", 50)
        self.learning_rate = control_data.get("learning_rate", 0.01)

        if controller_type == "PID":
            from control import PID   
            k_p = control_data.get("k_p", 1.0)
            k_i = control_data.get("k_i", 1.0)
            k_d = control_data.get("k_d", 1.0)
            self.controller = PID(control_config)
            self.controller_params = jnp.array([k_p, k_i, k_d], dtype=jnp.float32)  
        elif controller_type == "NN":
            from control import NN
            self.controller = NN(control_config)
            self.controller_params = self.controller.get_trainable_params()  
        else:
            raise ValueError("Invalid controller type. Use 'PID' or 'NN'.")

        self.error_history = []
        self.mse_history = []

        self.optimizer = optax.sgd(self.learning_rate)
        self.opt_state = self.optimizer.init(self.controller_params)

    def run_simulation(self):
        for epoch in range(self.epochs):
            self.plant.reset_plant()  

            if hasattr(self.controller, 'reset'):
                self.controller.reset()
            error_list = []
            for t in range(self.timesteps):
                error = self.plant.compute_error()
                control_signal = self.controller.control_signal(error)
                self.plant.update_state(control_signal)
                error_list.append(error)
                
            mse = self.loss_function(self.controller_params, error_list)
            self.mse_history.append(mse)

            gradients = self.compute_gradients(error_list)
            self.update_parameters(gradients)
            print(f"Epoch {epoch+1}/{self.epochs} - MSE: {mse:.5f}")

        print("Training complete!")
        
    def loss_function(self, params, error_list):
        return jnp.mean(jnp.array(error_list) ** 2)
    
    def compute_gradients(self, error_list):
        grad_fn = jax.grad(lambda params: self.loss_function(params, error_list))
        return grad_fn(self.controller_params)
    
    def update_parameters(self, gradients):
        updates, self.opt_state = self.optimizer.update(gradients, self.opt_state)
        self.controller_params = optax.apply_updates(self.controller_params, updates)

        if hasattr(self.controller, 'params'):
            self.controller.params = self.controller_params
        print(f"Updated Params: {self.controller_params}")
        
