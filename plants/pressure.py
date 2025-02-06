# pressure.py
import json
import jax
import jax.numpy as jnp
from .plant import Plant

class Pressure(Plant):
    def __init__(self, file_name):
        super().__init__(file_name)
        params = self.additional_parameters
        self.temperature = jnp.array(params.get("temperature", 300.0), dtype=jnp.float32)
        self.volume = jnp.array(params.get("volume", 10.0), dtype=jnp.float32)
    
    def reset_state(self):
        return self.initial_state  
    
    def compute_error(self, state):
        return self.target_state - state
    
    def update_state(self, state, control_signal, key):
        leak = self.temperature / self.volume
        new_state = state + control_signal - leak
        new_state = jnp.maximum(new_state, 0.0)
        return new_state, key

