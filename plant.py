import random
import json
import jax
from abc import ABC, abstractmethod

class Plant(ABC):

    def __init__(self, file_name):

        with open(file_name, "r") as config:
            self.config_parameters = json.load(config)

        global_parameters = self.config_parameters.get("parameters",{}) 
        #Subclas are gonna require "additional_parameters"

        self.initial_state = global_parameters.get("initial_state", 0)
        self.current_state = global_parameters.get("initial_state", 0)
        self.target_state = global_parameters.get("target_state", 0)
        self.disturbance_range = global_parameters.get("disturbance_range", (-0.01, 0.01))
        self.disturbance = 0 
        self.additional_parameters = self.config_parameters.get("additional_parameters", 0)

    @abstractmethod
    def update_state(self, control_signal):
        pass
        
    def compute_error(self):
        return self.target_state - self.current_state

    def generate_disturbance(self):
        self.disturbance = jax.random.uniform(*self.disturbance_range)
        return self.disturbance 

    def reset_plant(self):
        self.current_state = self.initial_state
