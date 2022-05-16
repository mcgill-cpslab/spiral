from tensorflow.python.ops import math_ops
from tensorflow.keras.layers import Layer, MultiHeadAttention
import tensorflow as tf

class StructuralAttentionLayer(Layer):
    def __init__(self, input_dim, output_dim, n_heads, attn_drop, ffd_drop, act=tf.nn.elu, residual=False,
                 bias=True, sparse_inputs=False, **kwargs):
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

    def call(self, g, h):
        self.n_calls += 1
        x = h
        adj = g.adj()
        attentions = []
        reuse_scope = None
        for j in range(self.n_heads):
            if self.n_calls > 1:
                reuse_scope = True

            attentions.append(self.sp_attn_head(x, adj_mat=adj, in_sz=self.input_dim,
                                                out_sz=self.output_dim // self.n_heads, activation=self.act,
                                                in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=self.residual,
                                                layer_str="l_{}_h_{}".format(self.name, j),
                                                sparse_inputs=self.sparse_inputs,
                                                reuse_scope=reuse_scope))

        h = tf.concat(attentions, axis=-1)
        return h

    @staticmethod
    def leaky_relu(features, alpha=0.2):
        return math_ops.maximum(alpha * features, features)

    def sp_attn_head(self, seq, in_sz, out_sz, adj_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False,
                     layer_str="", sparse_inputs=False, reuse_scope=None):
        """ Sparse Attention Head for the GAT layer. Note: the variable scope is necessary to avoid
        variable duplication across snapshots"""

        with tf.compat.v1.variable_scope('struct_attn', reuse=reuse_scope):
            if sparse_inputs:
                weight_var = tf.get_variable("layer_" + str(layer_str) + "_weight_transform", shape=[in_sz, out_sz],
                                             dtype=tf.float32)
                seq_fts = tf.expand_dims(tf.sparse_tensor_dense_matmul(seq, weight_var), axis=0)  # [N, F]
            else:
                seq = tf.reshape(seq,[1, seq.shape[0], -1])
                seq_fts = tf.compat.v1.layers.conv1d(seq, out_sz, 1, use_bias=False,
                                           name='layer_' + str(layer_str) + '_weight_transform', reuse=reuse_scope)

            # Additive self-attention.
            f_1 = tf.compat.v1.layers.conv1d(seq_fts, 1, 1, name='layer_' + str(layer_str) + '_a1', reuse=reuse_scope)
            f_2 = tf.compat.v1.layers.conv1d(seq_fts, 1, 1, name='layer_' + str(layer_str) + '_a2', reuse=reuse_scope)
            f_1 = tf.reshape(f_1, [-1, 1])  # [N, 1]
            f_2 = tf.reshape(f_2, [-1, 1])  # [N, 1]

            logits = tf.compat.v1.sparse_add(adj_mat * f_1, adj_mat * tf.transpose(f_2))  # adj_mat is [N, N] (sparse)

            leaky_relu = tf.SparseTensor(indices=logits.indices,
                                         values=self.leaky_relu(logits.values),
                                         dense_shape=logits.dense_shape)
            coefficients = tf.compat.v1.sparse_softmax(leaky_relu)  # [N, N] (sparse)

            if coef_drop != 0.0:
                coefficients = tf.SparseTensor(indices=coefficients.indices,
                                               values=tf.nn.dropout(coefficients.values, 1.0 - coef_drop),
                                               dense_shape=coefficients.dense_shape)  # [N, N] (sparse)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)  # [N, D]
            seq_fts = tf.squeeze(seq_fts)
            values = tf.compat.v1.sparse_tensor_dense_matmul(coefficients, seq_fts)
            values = tf.reshape(values, [-1, out_sz])
            values = tf.compat.v1.expand_dims(values, axis=0)
            ret = values  # [1, N, F]

            if residual:
                residual_wt = tf.get_variable("layer_" + str(layer_str) + "_residual_weight", shape=[in_sz, out_sz],
                                              dtype=tf.float32)
                if sparse_inputs:
                    ret = ret + tf.expand_dims(tf.sparse_tensor_dense_matmul(seq, residual_wt),
                                               axis=0)  # [N, F] * [F, D] = [N, D].
                else:
                    ret = ret + conv1d(seq, out_sz, 1, use_bias=False,
                                                 name='layer_' + str(layer_str) + '_residual_weight', reuse=reuse_scope)
            return tf.squeeze(activation(ret))
