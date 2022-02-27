import torch
import numpy as np

# Pulled straight from the IBP-Pytorch repo - NOT my work
def tensor_from(*args):
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]

    tensor_parts = []

    for part in args:
        if part is None:
            continue
        if isinstance(part, (int, float, complex)):
            part = torch.tensor([part]).float()
        if isinstance(part, np.number):
            part = torch.from_numpy(np.array([part])).float()
        if isinstance(part, np.ndarray):
            part = torch.from_numpy(part).float()

        if not isinstance(part, torch.Tensor):
            raise TypeError("This value has wrong type {}: {}".format(type(part), part))

        tensor_parts.append(part)

    for part in tensor_parts:
        print(part.shape)
    return torch.cat(tensor_parts)