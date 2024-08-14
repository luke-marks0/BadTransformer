import numpy as np
from typing import Dict


class AdamW:
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, 
                 epsilon: float = 1e-8, weight_decay: float = 0.01):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.m: Dict[str, Dict[str, np.ndarray]] = {}
        self.v: Dict[str, Dict[str, np.ndarray]] = {}

    def step(self, model, learning_rate: float):
        self.t += 1
        parameters = model.get_parameters()

        if not self.m:
            self.m = {layer: {name: np.zeros_like(param) 
                              for name, param in params.items()}
                      for layer, params in parameters.items()}
            self.v = {layer: {name: np.zeros_like(param) 
                              for name, param in params.items()}
                      for layer, params in parameters.items()}

        for layer_name, layer_params in parameters.items():
            for param_name, param in layer_params.items():
                grad = getattr(getattr(model, f"{layer_name}_layer"), f"d{param_name}")
                
                self.m[layer_name][param_name] = (self.beta1 * self.m[layer_name][param_name] + 
                                                  (1 - self.beta1) * grad)
                self.v[layer_name][param_name] = (self.beta2 * self.v[layer_name][param_name] + 
                                                  (1 - self.beta2) * (grad ** 2))

                m_hat = self.m[layer_name][param_name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[layer_name][param_name] / (1 - self.beta2 ** self.t)

                update = m_hat / (np.sqrt(v_hat) + self.epsilon)

                if 'b' not in param_name:
                    update += self.weight_decay * param

                setattr(getattr(model, f"{layer_name}_layer"), param_name, 
                        param - learning_rate * update)
