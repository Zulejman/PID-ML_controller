class Controller:

    def __init__(self, k_p, k_i, k_d):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.prev_error = 0
        self.integral_sum = 0

class PidController(Controller):

    def __init__(self, k_p, k_i, k_d):
        super().__init__(k_p, k_i, k_d)
    
    def proportion(self, k_p, error):
        return self.k_p * error 

    def integral(self, k_i, error):
        return self.k_i * error
    
    def derivate(self, k_d, error):
        derivate_value = self.k_d * (error - self.perv_error)
        self.prev_error = error
        return derivate_value

    def control_signal(self, error):
        P = self.proportion(error)
        I = self.integral(error)
        D = self.derivate(error)
        return P + I + D

#class nn_controller(controller):
