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
        self.params = jnp.array([self.k_p, self.k_i, self.k_d], dtype=jnp.float32)

    def pid_step(self, ctrl_state, error, params):
        prev_error, integral_sum = ctrl_state
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

class NN(Controller):
    def __init__(self, control_file):
        super().__init__(control_file)
        self.num_layers = self.control_cfg.get("layers", 0)
        self.neurons = self.control_cfg.get("neurons", 10)
        self.activation_name = self.control_cfg.get("activation", "relu").lower()
        self.init_range = self.control_cfg.get("init_range", [-0.1, 0.1])

        if self.activation_name == "sigmoid":
            self.activation = jax.nn.sigmoid
        elif self.activation_name == "tanh":
            self.activation = jax.nn.tanh
        elif self.activation_name == "relu":
            self.activation = jax.nn.relu
        else:
            raise ValueError(f"Typo: {self.activation_name}")

        layers = []
        key = jax.random.PRNGKey(42)
        lower, upper = self.init_range

        def init_layer(key, d_in, d_out):
            key_W, key_b, key = jax.random.split(key, 3)
            W = jax.random.uniform(key_W, (d_in, d_out), minval=lower, maxval=upper)
            b = jax.random.uniform(key_b, (d_out,), minval=lower, maxval=upper)
            return (W, b), key

        d_in = 3  
        if self.num_layers == 0:
            (W, b), key = init_layer(key, d_in, 1)
            layers.append((W, b))
        else:
            (W, b), key = init_layer(key, d_in, self.neurons)
            layers.append((W, b))
            for _ in range(self.num_layers - 1):
                (W, b), key = init_layer(key, self.neurons, self.neurons)
                layers.append((W, b))
            (W, b), key = init_layer(key, self.neurons, 1)
            layers.append((W, b))

        self.params = layers

    def nn_forward(self, x, params=None):
        if params is None:
            params = self.params
        out = x
        if len(params) == 1:
            W, b = params[0]
            y = jnp.dot(out, W) + b[0]
            return jnp.squeeze(y) 
        else:
            for layer in params[:-1]:
                W, b = layer
                out = jnp.dot(out, W) + b
                out = self.activation(out)
            W, b = params[-1]
            y = jnp.dot(out, W) + b[0]
            return jnp.squeeze(y)  

    def pid_step(self, ctrl_state, error, params):
        prev_error, integral_sum = ctrl_state
        new_integral = integral_sum + error
        derivative = error - prev_error
        error_vector = jnp.array([error, new_integral, derivative], dtype=jnp.float32)
        control = self.nn_forward(error_vector, params)
        new_ctrl_state = (error, new_integral)
        return control, new_ctrl_state

    def get_trainable_params(self):
        return self.params

    def reset(self):
        pass

