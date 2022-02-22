"""pytorch specific common functions."""

import torch as th
from numpy import ndarray
from typing import List

def _to_tensor(arr: ndarray) -> th.Tensor:
    """Convert a numpy array to torch tensor."""
    return th.from_numpy(arr)


def nt_layers_list() -> th.nn.ModuleList:
    return th.nn.ModuleList()


def reshape_tensor(h: th.Tensor, new_shape:List[int]) -> th.Tensor:
    return th.reshape(h,new_shape)
