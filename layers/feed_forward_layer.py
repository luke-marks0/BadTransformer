import numpy as np
from typing import Dict
from numpy.typing import NDArray


class FeedForwardLayer:
    def __init__(self, d_model: int, d_ff: int):
        self.w1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.b1 = np.zeros(d_ff)
        self.w2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        self.x = x
        self.z1 = np.einsum('bsd,df->bsf', x, self.w1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        output = np.einsum('bsf,fd->bsd', self.a1, self.w2) + self.b2
        return output

    def backward(self, dZ: NDArray[np.float64]) -> NDArray[np.float64]:
        self.dw2 = np.einsum('bsd,bsf->fd', dZ, self.a1)
        self.db2 = np.sum(dZ, axis=(0, 1))
        da1 = dZ @ self.w2.T
        dz1 = da1 * (self.z1 > 0)
        self.dw1 = np.einsum('bsd,bsf->df', self.x, dz1)
        self.db1 = np.sum(dz1, axis=(0, 1))
        return dz1 @ self.w1.T

    def get_parameters(self) -> Dict[str, NDArray[np.float64]]:
        return {
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2
        }
