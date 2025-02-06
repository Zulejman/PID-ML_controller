import json 
import jax.numpy as jnp

class Controller:

    def __init__(self, control_file):

        with open(control_file, "r") as ccfg:
            self.control_cfg = json.load(ccfg)

        self.k_p = self.control_cfg.get("k_p", 1)
        self.k_i = self.control_cfg.get("k_i", 1)
        self.k_d = self.control_cfg.get("k_d", 1)
        self.epochs = self.control_cfg.get("epochs", 10)
        self.step = self.control_cfg.get("steps", 5)
        self.learning_rate = self.control_cfg.get("learning_rate", 0.01)

class PID(Controller):

    def __init__(self, control_file):
        super().__init__(control_file)
        self.params = jnp.array([self.k_p, self.k_i, self.k_d], dtype=jnp.float32)
        self.prev_error = 0 
        self.integral_sum = 0
    
    def control_signal(self, error):
        k_p, k_i, k_d = self.params
        P = k_p * error
        self.integral_sum += error
        I = k_i * self.integral_sum
        D = k_d * (error - self.prev_error)
        self.prev_error = error
        print(f"P: {P}, I: {I}, D: {D}")
        return P + I + D

    def reset(self):
        self.prev_error = 0.0
        self.integral_sum = 0.0 

class NN(Controller):

    def __init__(self, control_file):
        super().__init__(control_file)

