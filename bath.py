from plant import Plant
import jax
import jax.numpy as jnp

class Bathtub(Plant):

    g = 9.8

    def __init__(self, file_name):
        super().__init__(file_name)
        bath_params = self.additional_parameters
        self.cross_section = jnp.array(bath_params.get("cross_section", 1.0), dtype=jnp.float32)
        self.drain = jnp.array(bath_params.get("drain", 0.01), dtype=jnp.float32)

    def give_velocity(self, state):
        return jnp.sqrt(2 * self.g * state)

    def flow_rate(self, state):
        return self.drain * self.give_velocity(state)

    def update_state(self, state, control_signal, key):
        key, subkey = jax.random.split(key)
        disturbance = jax.random.uniform(
            subkey,
            shape=(),
            minval=self.disturbance_range[0],
            maxval=self.disturbance_range[1]
        )
        volume_change = control_signal + disturbance - self.flow_rate(state)
        dh = volume_change / self.cross_section
        new_state = jnp.maximum(state + dh, 0.0)
        return new_state, key
