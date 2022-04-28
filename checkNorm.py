import numpy as np


def normalizeData(alist):
    mean = np.mean(alist, axis=0)
    std = np.std(alist, axis=0)
    normalized = (alist - mean) / std
    return normalized


data = [1, 9, 23, 5, 6, 8]
print(normalizeData(data))
