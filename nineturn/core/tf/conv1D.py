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
"""Tensorflow based conv1d layer."""

from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Conv1D, Layer

from nineturn.core.commonF import to_tensor
from nineturn.dtdg.models.decoder.tf.sequentialDecoder.baseModel import BaseModel


class Conv1d(Layer):
    """1-D convolutional layer for decoder"""

    def __init__(self, input_dim: int, out_dim: int, window_size: int, **kwargs):
        """Create a tsa layer.
        Args:
             in_dim:  int, input_dimension
             out_dim: int, output_dimension
        """
        super().__init__(name='Conv1dLayer')
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.window_size = window_size
        self.base_model = [Conv1D(out_dim, window_size, **kwargs) for i in range(window_size)]

    def call(self, input_window: Tensor) -> Tensor:
        slides = [self.base_model[i](input_window) for i in range(self.window_size)]
        result = tf.concat(slides, -2)
        return result
