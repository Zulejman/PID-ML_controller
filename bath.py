from plant import Plant

class Bathtub(Plant):
    G_C = 9.8

    def __init__(self, file_name):
        super().__init__(file_name)
        plant_parameters = self.config_parameters.get("additional_parameters",{}) 
        self.cross_section = plant_parameters.get("cross_section", 0)
        self.drain = plant_parameters.get("drain", 0)

    def update_state(self, control_signal):
        self.current_state = self.initial_state - self.heigth_change(control_signal)
        return self.current_state 

    def give_velocity(self):
        return (2 * self.G_C * self.current_state) ** 0.5

    def flow_rate(self):
        return self.drain * self.give_velocity()

    def volume_change(self, control_signal):
        return control_signal + self.generate_disturbance() - self.flow_rate() 
    def heigth_change(self, control_signal):
        return (self.volume_change(control_signal)) / self.current_state

    """ 
    Volume is equal to area times heigth: V = A * H
    So, the new heigth is: H = V / A

    In the json file height should be moved from additional_parameters to
    parameters. This should also be changed in the resto of the code.

    """

new_plant = Bathtub("bath_cfg.json")
print(new_plant.heigth_change(5))
