import json
import importlib
import jax
import jax.numpy as jnp
import optax

class CONSYS:
    def __init__(self, plant_config, control_config):
        # Load plant configuration.
        with open(plant_config, "r") as file:
            plant_data = json.load(file)
        # Load control configuration.
        with open(control_config, "r") as file:
            control_data = json.load(file)

        # Dynamically import the plant class.
        plant_module_name = plant_data.get("module", "plant")
        plant_class_name = plant_data.get("plant_class", "Bathtub")
        plant_module = importlib.import_module(plant_module_name)
        plant_class = getattr(plant_module, plant_class_name)
        # Create the plant instance.
        self.plant = plant_class(plant_config)

        # Set up training parameters.
        controller_type = control_data.get("controller", "PID")
        self.epochs = control_data.get("epochs", 100)
        self.timesteps = control_data.get("steps", 50)
        self.learning_rate = control_data.get("learning_rate", 0.01)

        # Create the controller instance.
        if controller_type == "PID":
            from control import PID
            self.controller = PID(control_config)
            # The trainable parameters (e.g., PID gains) are stored in self.controller.params.
            self.controller_params = self.controller.params
        elif controller_type == "NN":
            from control import NN
            self.controller = NN(control_config)
            self.controller_params = self.controller.get_trainable_params()
        else:
            raise ValueError("Invalid controller type. Use 'PID' or 'NN'.")

        # Set up the optimizer.
        self.optimizer = optax.sgd(self.learning_rate)
        self.opt_state = self.optimizer.init(self.controller_params)

        # Create an initial PRNG key.
        self.key = jax.random.PRNGKey(0)

        # Set up lists to store the MSE and parameter history per epoch.
        self.mse_history = []
        self.param_history = []  # This will store a list of [k_p, k_i, k_d] for each epoch.

    def run_simulation(self):
        """
        Runs a fully functional simulation loop (jitted) so that the computed error array—and hence
        the loss—is a differentiable function of the controller parameters.
        """
        @jax.jit
        def simulation_loop(controller_params, init_state, init_ctrl_state, key):
            def step_fn(carry, _):
                state, ctrl_state, key = carry
                # Compute error = target_state - current state.
                error = self.plant.compute_error(state)
                # Compute control and update the controller state.
                control, new_ctrl_state = self.controller.pid_step(ctrl_state, error, controller_params)
                # Update the plant state (and PRNG key).
                new_state, new_key = self.plant.update_state(state, control, key)
                return (new_state, new_ctrl_state, new_key), error

            # Unroll the simulation loop using jax.lax.scan.
            init_carry = (self.plant.reset_state(), (0.0, 0.0), key)
            (_, _, _), error_array = jax.lax.scan(step_fn, init_carry, jnp.arange(self.timesteps))
            return error_array

        @jax.jit
        def loss_fn(controller_params):
            init_state = self.plant.reset_state()
            init_ctrl_state = (0.0, 0.0)  # (prev_error, integral_sum)
            error_array = simulation_loop(controller_params, init_state, init_ctrl_state, self.key)
            return jnp.mean(error_array ** 2)

        # JIT the gradient function.
        grad_fn = jax.jit(jax.grad(loss_fn))

        # Training loop.
        for epoch in range(self.epochs):
            mse = loss_fn(self.controller_params)
            self.mse_history.append(float(mse))
            grads = grad_fn(self.controller_params)
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.controller_params = optax.apply_updates(self.controller_params, updates)
            # Update the controller's internal parameter copy if it exists.
            if hasattr(self.controller, 'params'):
                self.controller.params = self.controller_params

            # Record the current PID parameters (convert to Python list).
            self.param_history.append(self.controller_params.tolist())

            # Additional info: norm of gradient.
            grad_norm = jnp.linalg.norm(grads)
            print(f"Epoch {epoch+1}/{self.epochs} - MSE: {mse:.5f}, Params: {self.controller_params}, Grad Norm: {grad_norm:.5f}")

        print("Training complete!")

