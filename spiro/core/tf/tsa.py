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
"""Tensorflow based temporal self-attention layer."""

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Layer, MultiHeadAttention


class TSA(Layer):
    """Temporal self-attention layer."""

    def __init__(self, out_dim: int, num_heads: int, score: str = 'dot_product', **kwargs):
        """Create a tsa layer.

        Args:
             in_dim:  int, input_dimension
             out_dim: int, output_dimension
             num_heads: int, number of attention heads
             score: str, score function, currently only 'dot_product' is supported
        """
        super().__init__(name='TSALayer_' + score.upper())
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.base_model = MultiHeadAttention(num_heads=num_heads, key_dim=out_dim, **kwargs)

    def build(self, input_shape):
        """Initiate weights."""
        self.wq = self.add_weight(shape=(input_shape[-1], self.out_dim), initializer='uniform', trainable=True)

        self.wk = self.add_weight(shape=(input_shape[-1], self.out_dim), initializer='uniform', trainable=True)

        self.wv = self.add_weight(shape=(input_shape[-1], self.out_dim), initializer='uniform', trainable=True)
        super().build(input_shape)

    def call(self, query: Tensor, key: Tensor, value=None) -> Tensor:
        """Forward function.

        Args:
            query: A Tensor with shape (num_nodes, window_size, feature_size)
            key: The keys to align with query. Usually is the same as the input query in self-attention
            value: usually the same as key. Default to key

        Return:
            new_sequence with shape (num_nodes, window_size, out_dim)
        """
        if value is None:
            value = key
        Q = tf.matmul(query, self.wq)
        K = tf.matmul(key, self.wk)
        V = tf.matmul(value, self.wv)
        return self.base_model(Q, K, V)

    def compute_output_shape(self, input_shape):
        """Compute the output shape."""
        return (input_shape[0], input_shape[1], self.out_dim)


class DysatPtsa(Layer):
    """Positional Temporal Self-Attention layer for Dysat."""

    def __init__(
        self,
        input_dim,
        n_heads,
        num_time_steps,
        attn_drop=0.2,
        residual=False,
        bias=True,
        use_position_embedding=True,
        **kwargs,
    ):
        """Initialize a PTSA layer.

        Args:
            input_dim: the dimension of the input
            n_heads: number of attention heads
            num_time_steps: the length of input sliding window
            attn_drop: dropout rate of the attention layer in training
            residual: default is False
            bias: default is True
            use_position_embedding: default is True
        """
        super().__init__(name='Dysat_PTSALayer_')
        self.bias = bias
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.attn_wts_means = []
        self.attn_wts_vars = []
        self.residual = residual
        self.input_dim = input_dim
        self.attn_drop = attn_drop

        xavier_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.name + '_vars'):
            self.vars['position_embeddings'] = tf.get_variable(
                'position_embeddings', dtype=tf.float32, shape=[self.num_time_steps, input_dim], initializer=xavier_init
            )  # [T, F]

            self.vars['Q_embedding_weights'] = tf.get_variable(
                'Q_embedding_weights', dtype=tf.float32, shape=[input_dim, input_dim], initializer=xavier_init
            )  # [F, F]
            self.vars['K_embedding_weights'] = tf.get_variable(
                'K_embedding_weights', dtype=tf.float32, shape=[input_dim, input_dim], initializer=xavier_init
            )  # [F, F]
            self.vars['V_embedding_weights'] = tf.get_variable(
                'V_embedding_weights', dtype=tf.float32, shape=[input_dim, input_dim], initializer=xavier_init
            )  # [F, F]

    def call(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]."""
        # 1: Add position embeddings to input
        position_inputs = tf.tile(tf.expand_dims(tf.range(self.num_time_steps), 0), [tf.shape(inputs)[0], 1])
        temporal_inputs = inputs + tf.nn.embedding_lookup(self.vars['position_embeddings'], position_inputs)
        # 2: Query, Key based multi-head self attention.
        q = tf.tensordot(temporal_inputs, self.vars['Q_embedding_weights'], axes=[[2], [0]])  # [N, T, F]
        k = tf.tensordot(temporal_inputs, self.vars['K_embedding_weights'], axes=[[2], [0]])  # [N, T, F]
        v = tf.tensordot(temporal_inputs, self.vars['V_embedding_weights'], axes=[[2], [0]])  # [N, T, F]

        # 3: Split, concat and scale.
        q_ = tf.concat(tf.split(q, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]
        k_ = tf.concat(tf.split(k, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]
        v_ = tf.concat(tf.split(v, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]

        outputs = tf.matmul(q_, tf.transpose(k_, [0, 2, 1]))  # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)

        # 4: Masked (causal) softmax to compute attention weights.

        diag_val = tf.ones_like(outputs[0, :, :])  # [T, T]
        tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [T, T]
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # [hN, T, T]
        padding = tf.ones_like(masks) * (-(2 ** 32) + 1)
        outputs = tf.where(tf.equal(masks, 0), padding, outputs)  # [h*N, T, T]
        outputs = tf.nn.softmax(outputs)  # Masked attention.
        self.attn_wts_all = outputs

        # 5: Dropout on attention weights.
        outputs = tf.layers.dropout(outputs, rate=self.attn_drop)
        outputs = tf.matmul(outputs, v_)  # [hN, T, C/h]
        split_outputs = tf.split(outputs, self.n_heads, axis=0)
        outputs = tf.concat(split_outputs, axis=-1)

        # Optional: residual
        if self.residual:
            outputs += temporal_inputs
        return outputs
