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


def from_numpy_img(array, if_resize=True, target_shape=(256, 256)):
    # array: (B, H, W, C)
    array = (array / 255.).astype(np.float32)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    if if_resize:
        resizer = Resize(target_shape)
        tensor = resizer(tensor)

    return tensor.to(device).to(torch.float32)
