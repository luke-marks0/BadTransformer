import numpy as np
from typing import List, Dict
from layers.attention_layer import AttentionLayer
from layers.feed_forward_layer import FeedForwardLayer
from adamw import AdamW
from numpy.typing import NDArray


class Transformer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attention_layer = AttentionLayer(d_model, num_heads)
        self.feed_forward_layer = FeedForwardLayer(d_model, d_ff)
        self.optimizer = AdamW()

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        attention_output = self.attention_layer.forward(x)
        output = self.feed_forward_layer.forward(attention_output)
        return output

    def backward(self, dZ: NDArray[np.float64], x: NDArray[np.float64]) -> None:
        d_ff = self.feed_forward_layer.backward(dZ)
        self.attention_layer.backward(d_ff, x)

    def get_parameters(self) -> Dict[str, Dict[str, NDArray[np.float64]]]:
        return {
            'attention': self.attention_layer.get_parameters(),
            'feed_forward': self.feed_forward_layer.get_parameters()
        }

    def train(self, X: List[NDArray[np.float64]], y: List[NDArray[np.float64]], 
              learning_rate: float, epochs: int, batch_size: int) -> None:
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                batch_X = np.array(X[i:i+batch_size])
                batch_y = np.array(y[i:i+batch_size])

                output = self.forward(batch_X)
                loss = np.mean((output - batch_y) ** 2)
                print(loss)
                dZ = 2 * (output - batch_y) / output.size

                self.backward(dZ, batch_X)
                self.optimizer.step(self, learning_rate)
