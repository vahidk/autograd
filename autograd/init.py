import numpy as np


def kaiming(shape, mode="fan_in", gain=1.0, distribution="uniform"):
    if len(shape) >= 2:
        in_channels = shape[-2]
        out_channels = shape[-1]
    else:
        raise ValueError("Shape must have at least 2 dimensions")

    if len(shape) > 2:
        receptive_field_size = np.prod(shape[:-2])
        fan_in = in_channels * receptive_field_size
        fan_out = out_channels * receptive_field_size
    else:
        fan_in = in_channels
        fan_out = out_channels

    if mode == "fan_in":
        fan = fan_in
    elif mode == "fan_out":
        fan = fan_out
    else:
        raise ValueError("Mode must be either 'fan_in' or 'fan_out'")

    std = gain / np.sqrt(fan)

    if distribution == "normal":
        weights = np.random.normal(0.0, std, shape)
    elif distribution == "uniform":
        bound = np.sqrt(3.0) * std
        weights = np.random.uniform(-bound, bound, shape)
    else:
        raise ValueError("Distribution must be either 'normal' or 'uniform'")

    return weights
