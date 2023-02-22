#! /usr/bin/env python3

import numpy as np

def qsin(data, ssb):
    """
    Sine-squared apodization.
    """

    if ssb == 0 or ssb == 1:
        off = 0
    else:
        off = 1/ssb
    end = 1
    size = data.shape[-1]
    apod = np.power(np.sin(np.pi * off + np.pi * (end - off) * np.arange(size) / (size)).astype(data.dtype), 2).astype(data.dtype)
    return apod * data
