# Copyright 2022 The Nine Turn Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""pytorch specific common functions."""

from typing import List

import torch
from numpy import ndarray
from torch import Tensor


def _to_tensor(arr: ndarray) -> Tensor:
    """Convert a numpy array to torch tensor."""
    return torch.from_numpy(arr)


def nt_layers_list() -> torch.nn.ModuleList:
    """Create module list to store layers."""
    return torch.nn.ModuleList()


def reshape_tensor(h: torch.Tensor, new_shape: List[int]) -> Tensor:
    """Reshape input tensor to new_shape.

    Args:
        h: Tensor to reshape
        new_shape: the new shape after reshape

    Return:
        new Tensor with new_shape
    """
    return torch.reshape(h, new_shape)
