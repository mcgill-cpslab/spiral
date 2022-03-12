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
"""Tensorflow specific common functions."""
from typing import List

import tensorflow as tf
from numpy import ndarray


def _to_tensor(arr: ndarray, dtype=None) -> tf.Tensor:
    """Convert a numpy array to tensorflow tensor."""
    return tf.constant(arr, dtype=dtype)


def nt_layers_list() -> List:
    """Create a list to store layers."""
    return []


def reshape_tensor(h: tf.Tensor, new_shape: List[int]) -> tf.Tensor:
    """Reshape input tensor to new_shape.

    Args:
        h: Tensor to reshape
        new_shape: the new shape after reshape

    Return:
        new Tensor with new_shape
    """
    return tf.reshape(h, new_shape)
