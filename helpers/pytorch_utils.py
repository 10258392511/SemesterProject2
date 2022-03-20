import torch
import numpy as np

from torchvision.transforms import Resize

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def from_numpy(array):
    if array.dtype in [np.float64, np.float32]:
        dtype = torch.float32
    elif array.dtype in [np.int64, np.int32]:
        dtype = torch.long
    else:
        array = array.astype(np.float32)
        dtype = torch.float32
    return torch.from_numpy(array).to(device).to(dtype)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

