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
"""Tensorflow based structural attention layer, specific for DySat."""

import tensorflow as tf
from dgl import DGLGraph
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import math_ops


class StructuralAttentionLayer(Layer):
    """StructuralAttentionLayer for both the encoder and decoder in DySat."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_heads: int,
        attn_drop: float,
        ffd_drop: float,
        act=tf.nn.elu,
        residual: bool = False,
        bias: bool = True,
        sparse_inputs: bool = False,
        **kwargs,
    ):
        """Create a structural attention layer.

        Args:
             input_dim:  int, input_dimension
             output_dim: int, output_dimension
             n_heads: int, number of attention heads
             attn_drop: float, dropout possibility for the attention layer
             ffd_drop: float, dropout possibility for the feedforward layer
             act: activation function.
             residual: boolean, whether to use residual
             bias: boolean, use bias or not
             sparse_inputs: boolean, the input is a sparse tensor
             **kwargs: other tf.nn.layer supported arguments
        """
        super(StructuralAttentionLayer, self).__init__(**kwargs)
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.act = act
        self.bias = bias
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = residual
        self.sparse_inputs = sparse_inputs
        self.n_calls = 0

    def call(self, g: DGLGraph, h: tf.Tensor) -> tf.Tensor:
        """The call function of this layer.

        Args:
            g: a DGLGraph object
            h: a tensor representing the node features
        Return:
            a tensor representing the new node features after message passing
        """
        self.n_calls += 1
        x = h
        adj = g.adj()
        attentions = []
        reuse_scope = None
        for j in range(self.n_heads):
            if self.n_calls > 1:
                reuse_scope = True

            attentions.append(
                self.sp_attn_head(
                    x,
                    adj_mat=adj,
                    in_sz=self.input_dim,
                    out_sz=self.output_dim // self.n_heads,
                    activation=self.act,
                    in_drop=self.ffd_drop,
                    coef_drop=self.attn_drop,
                    residual=self.residual,
                    layer_str="l_{}_h_{}".format(self.name, j),
                    sparse_inputs=self.sparse_inputs,
                    reuse_scope=reuse_scope,
                )
            )
        h = tf.concat(attentions, axis=-1)
        return h

    @staticmethod
    def leaky_relu(features: tf.Tensor, alpha: float = 0.2):
        """Leaky Relu.

        Args:
            features: the input node featurs
            alpha: float
        """
        return math_ops.maximum(alpha * features, features)

    def sp_attn_head(
        self,
        seq,
        in_sz,
        out_sz,
        adj_mat,
        activation,
        in_drop=0.0,
        coef_drop=0.0,
        residual=False,
        layer_str="",
        sparse_inputs=False,
        reuse_scope=None,
    ):
        """Sparse Attention Head for the GAT layer.

        The variable scope is necessary to avoid variable duplication across snapshots

        Args:
            seq: the input sequence,
            in_sz: input feature size,
            out_sz: output embeding size,
            adj_mat: the adjacency matrix,
            activation: activation function,
            in_drop: dropout for input feature,
            coef_drop: dropout for coefficients,
            residual: use residual or not,
            layer_str: name for the layer,
            sparse_inputs: input tensor is sparse or not,
            reuse_scope: reuse the same scope,
        """
        with tf.compat.v1.variable_scope('struct_attn', reuse=reuse_scope):
            if sparse_inputs:
                weight_var = tf.get_variable(
                    "layer_" + str(layer_str) + "_weight_transform", shape=[in_sz, out_sz], dtype=tf.float32
                )
                seq_fts = tf.expand_dims(tf.sparse_tensor_dense_matmul(seq, weight_var), axis=0)  # [N, F]
            else:
                seq = tf.reshape(seq, [1, seq.shape[0], -1])
                seq_fts = tf.compat.v1.layers.conv1d(
                    seq,
                    out_sz,
                    1,
                    use_bias=False,
                    name='layer_' + str(layer_str) + '_weight_transform',
                    reuse=reuse_scope,
                )

            # Additive self-attention.
            f_1 = tf.compat.v1.layers.conv1d(seq_fts, 1, 1, name='layer_' + str(layer_str) + '_a1', reuse=reuse_scope)
            f_2 = tf.compat.v1.layers.conv1d(seq_fts, 1, 1, name='layer_' + str(layer_str) + '_a2', reuse=reuse_scope)
            f_1 = tf.reshape(f_1, [-1, 1])  # [N, 1]
            f_2 = tf.reshape(f_2, [-1, 1])  # [N, 1]

            logits = tf.compat.v1.sparse_add(adj_mat * f_1, adj_mat * tf.transpose(f_2))  # adj_mat is [N, N] (sparse)

            leaky_relu = tf.SparseTensor(
                indices=logits.indices, values=self.leaky_relu(logits.values), dense_shape=logits.dense_shape
            )
            coefficients = tf.compat.v1.sparse_softmax(leaky_relu)  # [N, N] (sparse)

            if coef_drop != 0.0:
                coefficients = tf.SparseTensor(
                    indices=coefficients.indices,
                    values=tf.nn.dropout(coefficients.values, 1.0 - coef_drop),
                    dense_shape=coefficients.dense_shape,
                )  # [N, N] (sparse)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)  # [N, D]
            seq_fts = tf.squeeze(seq_fts)
            values = tf.compat.v1.sparse_tensor_dense_matmul(coefficients, seq_fts)
            values = tf.reshape(values, [-1, out_sz])
            values = tf.compat.v1.expand_dims(values, axis=0)
            ret = values  # [1, N, F]

            if residual:
                residual_wt = tf.get_variable(
                    "layer_" + str(layer_str) + "_residual_weight", shape=[in_sz, out_sz], dtype=tf.float32
                )
                if sparse_inputs:
                    ret = ret + tf.expand_dims(
                        tf.sparse_tensor_dense_matmul(seq, residual_wt), axis=0
                    )  # [N, F] * [F, D] = [N, D].
                else:
                    ret = ret + tf.compat.v1.layers.conv1d(
                        seq,
                        out_sz,
                        1,
                        use_bias=False,
                        name='layer_' + str(layer_str) + '_residual_weight',
                        reuse=reuse_scope,
                    )
            return tf.squeeze(activation(ret))
