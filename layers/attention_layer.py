import numpy as np
from typing import Dict, Optional
from numpy.typing import NDArray


class AttentionLayer:
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.w_q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.w_k = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.w_v = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.w_o = np.random.randn(d_model, d_model) / np.sqrt(d_model)

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        self.x = x
        batch_size, seq_len, _ = x.shape
        
        q = np.einsum('bsd,dh->bsh', x, self.w_q)
        k = np.einsum('bsd,dh->bsh', x, self.w_k)
        v = np.einsum('bsd,dh->bsh', x, self.w_v)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        self.v = v

        scores = np.einsum('bqhd,bkhd->bhqk', q, k) / np.sqrt(self.head_dim)
        
        self.attention_weights = self._softmax(scores)
        attention_output = np.einsum('bhqk,bkhd->bqhd', self.attention_weights, v)
        
        attention_output = attention_output.reshape(batch_size, seq_len, self.d_model)
        output = np.einsum('bsd,dd->bsd', attention_output, self.w_o)
        
        return output

    def backward(self, dZ: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
        batch_size, seq_len, _ = x.shape
        
        self.dw_o = np.einsum('bsd,bsh->dh', dZ, x)
        d_attention_output = np.einsum('bsd,dd->bsd', dZ, self.w_o.T).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        dv = np.einsum('bhqk,bqhd->bkhd', self.attention_weights, d_attention_output)
        d_scores = np.einsum('bqhd,bkhd->bhqk', d_attention_output, self.v)
        d_scores *= self.attention_weights
        d_scores -= self.attention_weights * np.sum(d_scores * self.attention_weights, axis=-1, keepdims=True)
        d_scores /= np.sqrt(self.head_dim)
        
        dq = np.einsum('bhqk,bkhd->bqhd', d_scores, self.v)
        dk = np.einsum('bhkq,bqhd->bkhd', d_scores, dq)

        self.dw_q = np.einsum('bsd,bsh->dh', x, dq.reshape(batch_size, seq_len, self.d_model))
        self.dw_k = np.einsum('bsd,bsh->dh', x, dk.reshape(batch_size, seq_len, self.d_model))
        self.dw_v = np.einsum('bsd,bsh->dh', x, dv.reshape(batch_size, seq_len, self.d_model))

        return np.einsum('bsd,dh->bsh', dZ, self.w_o.T)

    def get_parameters(self) -> Dict[str, NDArray[np.float64]]:
        return {
            'w_q': self.w_q,
            'w_k': self.w_k,
            'w_v': self.w_v,
            'w_o': self.w_o
        }

    @staticmethod
    def _softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
