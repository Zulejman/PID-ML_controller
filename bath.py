from plant import Plant
import jax.numpy as jnp

class Bathtub(Plant):

    G_C = 9.8

    def __init__(self, file_name):
        super().__init__(file_name)
        bath_parameters = self.config_parameters.get("additional_parameters",{}) 
        self.cross_section = bath_parameters.get("cross_section", 0)
        self.drain = bath_parameters.get("drain", 0)

    def update_state(self, control_signal):
        dh = self.height_change(control_signal)
        self.current_state = self.current_state + dh
        self.current_state = jnp.maximum(self.current_state, 0)
        return self.current_state 


    def give_velocity(self):
        return (2 * self.G_C * self.current_state) ** 0.5

    def flow_rate(self):
        return self.drain * self.give_velocity()

    def volume_change(self, control_signal):
        return control_signal + self.generate_disturbance() - self.flow_rate() 

    def height_change(self, control_signal):
        return (self.volume_change(control_signal)) / self.cross_section
