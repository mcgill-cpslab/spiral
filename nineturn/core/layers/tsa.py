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
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Layer, MultiHeadAttention
from tensorflow.keras.backend import sin, cos, dot, concatenate

from nineturn.core.commonF import to_tensor
from nineturn.dtdg.models.decoder.tf.sequentialDecoder.baseModel import BaseModel

periodic_functions = {'sin': sin, 'cos': cos}


class TSA(Layer):
    """Temporal self-attention layer."""
    def __init__(self,  out_dim:int,num_heads:int, score:str = 'dot_product', **kwargs):
        """Create a tsa layer.
        Args:
             in_dim:  int, input_dimension
             out_dim: int, output_dimension
             num_heads: int, number of attention heads
             score: str, score function, currently only 'dot_product' is supported
        """
        super().__init__(
            name='TSALayer_'+ score.upper()
        )
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.base_model = MultiHeadAttention(num_heads=num_heads, key_dim=out_dim, **kwargs)
        
    
    def build(self, input_shape):
        
        self.wq = self.add_weight(
            shape=(input_shape[-1], self.out_dim),
            initializer='uniform',
            trainable=True
        )
        
        self.wk = self.add_weight(
            shape=(input_shape[-1], self.out_dim),
            initializer='uniform',
            trainable=True
        )
        
        self.wv = self.add_weight(
            shape=(input_shape[-1], self.out_dim),
            initializer='uniform',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, query, key, value=None):
        """Forward function.
        Args:
            inputs: A Tensor with shape (num_nodes, window_size, feature_size)
        
        Return: 
            new_sequence with shape (num_nodes, window_size, out_dim)
        """
        if value is None:
            value = key
        Q = tf.matmul(query, self.wq)
        K = tf.matmul(key, self.wk)
        V = tf.matmul(value, self.wv)
        return self.base_model(Q,K,V)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.out_dim)
