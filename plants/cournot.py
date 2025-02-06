import json
import jax
import jax.numpy as jnp
from .plant import Plant

class Cournot(Plant):

    def __init__(self, file_name):
        super().__init__(file_name)
        params = self.additional_parameters
        self.pmax = jnp.array(params.get("pmax", 10.0), dtype=jnp.float32)  
        self.cm = jnp.array(params.get("cm", 0.1), dtype=jnp.float32)
        self.initial_q2 = jnp.array(params.get("q2_initial", 0.5), dtype=jnp.float32)

    def reset_state(self):
        q1 = self.initial_state  
        q2 = self.initial_q2    
        return (q1, q2)

    def compute_error(self, state):
        q1, q2 = state
        q_total = q1 + q2
        price = self.pmax - q_total
        profit = q1 * (price - self.cm)
        error = self.target_state - profit
        return error

    def update_state(self, state, control_signal, key):
        q1, q2 = state
        key, subkey = jax.random.split(key)
        disturbance = jax.random.uniform(
            subkey,
            shape=(),
            minval=self.disturbance_range[0],
            maxval=self.disturbance_range[1]
        )
        q1_new = q1 + control_signal
        q2_new = q2 + disturbance
        q1_new = jnp.clip(q1_new, 0.0, 1.0)
        q2_new = jnp.clip(q2_new, 0.0, 1.0)
        return (q1_new, q2_new), key

