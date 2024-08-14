import numpy as np
from transformer import Transformer
from global_normalization import global_normalization

d_model = 64
num_heads = 4
d_ff = 256
learning_rate = 1
batch_size = 32
epochs = 10

X = [np.random.randn(10, d_model) for _ in range(1000)]
y = [np.full_like(X[0], 1) for _ in range(1000)]

X = global_normalization(X)

transformer = Transformer(d_model, num_heads, d_ff)
transformer.train(X, y, learning_rate, epochs, batch_size)
