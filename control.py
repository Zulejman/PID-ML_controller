import json
import jax
import jax.numpy as jnp

class Controller:
    def __init__(self, control_file):
        with open(control_file, "r") as ccfg:
            self.control_cfg = json.load(ccfg)

        self.k_p = self.control_cfg.get("k_p", 1.0)
        self.k_i = self.control_cfg.get("k_i", 0.1)
        self.k_d = self.control_cfg.get("k_d", 0.01)
        self.epochs = self.control_cfg.get("epochs", 10)
        self.steps = self.control_cfg.get("steps", 5)
        self.learning_rate = self.control_cfg.get("learning_rate", 0.01)

class PID(Controller):
    def __init__(self, control_file):
        super().__init__(control_file)
        # Store the PID gains as a JAX array.
        self.params = jnp.array([self.k_p, self.k_i, self.k_d], dtype=jnp.float32)

    def pid_step(self, ctrl_state, error, params):
        """
        A functional PID step.

        Args:
            ctrl_state: A tuple (prev_error, integral_sum).
            error: The current error.
            params: A JAX array with PID gains [k_p, k_i, k_d].

        Returns:
            control: The computed control signal.
            new_ctrl_state: The updated controller state.
        """
        prev_error, integral_sum = ctrl_state
        # Use the passed-in params (NOT self.params).
        P = params[0] * error
        new_integral = integral_sum + error
        I = params[1] * new_integral
        D = params[2] * (error - prev_error)
        control = P + I + D
        new_ctrl_state = (error, new_integral)
        return control, new_ctrl_state

    def run(self, errors, init_state=(0.0, 0.0)):
        errors = jnp.array(errors, dtype=jnp.float32)

        def step_fn(state, error):
            new_state, control = self.pid_step(state, error, self.params)
            return new_state, control

        final_state, controls = jax.lax.scan(step_fn, init_state, errors)
        return controls

