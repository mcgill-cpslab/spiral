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

import torch as th
from numpy import ndarray


def _to_tensor(arr: ndarray) -> th.Tensor:
    """Convert a numpy array to torch tensor."""
    return th.from_numpy(arr)


def nt_layers_list() -> th.nn.ModuleList:
    """Create module list to store layers."""
    return th.nn.ModuleList()


def reshape_tensor(h: th.Tensor, new_shape: List[int]) -> th.Tensor:
    """Reshape input tensor to new_shape.

    Args:
        h: Tensor to reshape
        new_shape: the new shape after reshape

    Return:
        new Tensor with new_shape
    """
    return th.reshape(h, new_shape)
