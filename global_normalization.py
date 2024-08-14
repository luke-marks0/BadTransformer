import numpy as np
from numpy.typing import NDArray
from typing import List


def global_normalization(X: List[NDArray[np.float64]]) -> List[NDArray[np.float64]]:
    global_mean = np.mean([np.mean(x) for x in X])
    global_var = np.mean([np.var(x) for x in X] + 
                         [(np.mean(x) - global_mean)**2 for x in X])
    global_std = np.sqrt(global_var)
    
    X_normalized = [(x - global_mean) / global_std for x in X]
    
    return X_normalized
