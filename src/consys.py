import json
import importlib
import jax
import jax.numpy as jnp
import optax
from jax import tree_util

class CONSYS:

    
    def __init__(self, plant_config, control_config):

        with open(plant_config, "r") as file:
            plant_data = json.load(file)
        with open(control_config, "r") as file:
            control_data = json.load(file)

        plant_module_name = plant_data.get("module", "plants.plant")
        plant_class_name = plant_data.get("plant_class", "Bathtub")
        plant_module = importlib.import_module(plant_module_name)
        plant_class = getattr(plant_module, plant_class_name)
        self.plant = plant_class(plant_config)

        controller_type = control_data.get("controller", "PID")
        self.epochs = control_data.get("epochs", 100)
        self.timesteps = control_data.get("steps", 50)
        self.learning_rate = control_data.get("learning_rate", 0.01)

        if controller_type == "PID":
            from .control import PID
            self.controller = PID(control_config)
            self.controller_params = self.controller.params
        elif controller_type == "NN":
            from .control import NN
            self.controller = NN(control_config)
            self.controller_params = self.controller.get_trainable_params()
        else:
            raise ValueError("Invalid controller type. Use 'PID' or 'NN'.")

        self.optimizer = optax.sgd(self.learning_rate)
        self.opt_state = self.optimizer.init(self.controller_params)

        self.key = jax.random.PRNGKey(0)

        self.mse_history = []
        self.param_history = []  

    def run_simulation(self):
        @jax.jit
        def simulation_loop(controller_params, init_state, init_ctrl_state, key):
            def step_fn(carry, _):
                state, ctrl_state, key = carry
                error = self.plant.compute_error(state)
                control, new_ctrl_state = self.controller.pid_step(ctrl_state, error, controller_params)
                new_state, new_key = self.plant.update_state(state, control, key)
                return (new_state, new_ctrl_state, new_key), error

            init_carry = (self.plant.reset_state(), (0.0, 0.0), key)
            (_, _, _), error_array = jax.lax.scan(step_fn, init_carry, jnp.arange(self.timesteps))
            return error_array

        @jax.jit
        def loss_fn(controller_params):
            init_state = self.plant.reset_state()
            init_ctrl_state = (0.0, 0.0)  
            error_array = simulation_loop(controller_params, init_state, init_ctrl_state, self.key)
            return jnp.mean(error_array ** 2)

        grad_fn = jax.jit(jax.grad(loss_fn))

        for epoch in range(self.epochs):
            mse = loss_fn(self.controller_params)
            self.mse_history.append(float(mse))
            grads = grad_fn(self.controller_params)
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.controller_params = optax.apply_updates(self.controller_params, updates)
            if hasattr(self.controller, 'params'):
                self.controller.params = self.controller_params

            self.param_history.append(self.controller_params.tolist())

            grad_norm = jnp.linalg.norm(grads)
            print(f"Epoch {epoch+1}/{self.epochs} - MSE: {mse:.5f}, Params: {self.controller_params}, Grad Norm: {grad_norm:.5f}")

        print("Training complete!") 
