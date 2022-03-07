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
"""Tensorflow based time2vec encoding."""
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.backend import cos, sin
from tensorflow.keras.layers import Layer

periodic_functions = {'sin': sin, 'cos': cos}


class Time2Vec(Layer):
    """Time2Vec layer."""

    def __init__(self, kernel_size: int, feature_size: int, activation: str = 'sin'):
        """Create a time2vec layer.

        Args:
             kernel_size:  int, the length of time vector representation.
             feature_size: int, the number of input features
             activation: str, the periodic activation,one of 'sin' or 'cos'.
        """
        super().__init__(name='Time2VecLayer_' + activation.upper())

        self.kernel_size = kernel_size
        self.feature_size = feature_size
        self.activation = periodic_functions[activation]

    def build(self, input_shape):
        """Build function to initiate weights."""
        # While i = 0, linear projection
        self.wb = self.add_weight(shape=(1, self.feature_size), initializer='uniform', trainable=True)

        self.bb = self.add_weight(shape=(1, self.feature_size), initializer='uniform', trainable=True)

        # Else needs to pass the periodic activation
        self.wa = self.add_weight(
            shape=(1, self.feature_size, self.kernel_size - 1), initializer='uniform', trainable=True
        )

        self.ba = self.add_weight(
            shape=(self.feature_size, self.kernel_size - 1), initializer='uniform', trainable=True
        )

        super().build(input_shape)

    def call(self, inputs: Tensor) -> Tensor:
        """Forward function.

        Args:
            inputs: A Tensor with shape (batch_size, window_size)

        Return:
            time2vec encoding of the input time dimension with shape
                (batch_size, feature_size, length of time vector representation + 1)
        """
        inputs = tf.reshape(inputs, [inputs.shape[0], inputs.shape[1], 1])
        bias = tf.reshape(
            tf.matmul(inputs, self.wb) + self.bb, [inputs.shape[0], inputs.shape[1], self.feature_size, 1]
        )  # n w d 1
        pro = tf.reshape(
            tf.einsum('abc, cde-> abcde', inputs, self.wa), [inputs.shape[0], inputs.shape[1], self.feature_size, -1]
        )
        sums = pro + self.ba  # n w d k-1
        wgts = self.activation(sums)  # n w d k-1
        return tf.concat([bias, wgts], 3)

    def compute_output_shape(self, input_shape):
        """Compute the output shape."""
        return (input_shape[0][0], input_shape[1], self.kernel_size, self.feature_size)
