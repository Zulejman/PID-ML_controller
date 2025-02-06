import json
import jax.numpy as jnp
from abc import ABC, abstractmethod

class Plant(ABC):

    def __init__(self, file_name):

        with open(file_name, "r") as config:
            self.config_parameters = json.load(config)

        global_parameters = self.config_parameters.get("parameters",{}) 

        self.initial_state = jnp.array(global_parameters.get("initial_state", 0.0), dtype=jnp.float32)
        self.target_state = jnp.array(global_parameters.get("target_state", 0.0), dtype=jnp.float32)
        self.disturbance_range = tuple(global_parameters.get("disturbance_range", [-0.01, 0.01]))
        self.additional_parameters = self.config_parameters.get("additional_parameters", {})

    @abstractmethod
    def update_state(self, state, control_signal, key):
        pass
        
    def compute_error(self, state):
        return self.target_state - state 

    def reset_state(self):
        return self.initial_state
