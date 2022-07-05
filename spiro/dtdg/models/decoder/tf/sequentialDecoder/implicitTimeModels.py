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
"""Tensorflow based sequential decoder. Designed specially for dynamic graph learning."""
from typing import List, Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import RNN as TfRnn
from tensorflow.keras.layers import GRUCell, LSTMCell, SimpleRNNCell

from spiro.core.commonF import to_tensor
from spiro.core.errors import ValueError
from spiro.core.layers import TSA, Conv1d, Time2Vec
from spiro.core.types import nt_layers_list
from spiro.dtdg.models.decoder.tf.sequentialDecoder.baseModel import (
    NodeMemory,
    NodeTrackingFamily,
    RnnFamily,
    SlidingWindowFamily,
    _process_target_ids,
)
from spiro.dtdg.models.decoder.tf.simpleDecoder import SimpleDecoder


class LSTM(RnnFamily):
    """LSTM sequential decoder."""

    def __init__(
        self,
        input_d: int,
        hidden_d: int,
        n_nodes: int,
        n_layers: int,
        simple_decoder: SimpleDecoder,
        **kwargs,
    ):
        """Create a LSTM sequential decoder.

        Args:
            input_d: int, input dimension.
            hidden_d: int, number of hidden cells.
            n_nodes: int, number of nodes.
            n_layers: int, number of lstm layers.
            simple_decoder: an instance of SimpleDecoder.
        """
        RnnFamily.__init__(self, hidden_d, n_nodes, n_layers, simple_decoder)
        self.input_d = input_d
        self.base_model = TfRnn(
            [LSTMCell(hidden_d, **kwargs) for i in range(n_layers)],
            return_state=True,
            return_sequences=True,
            **kwargs,
        )
        self.memory_c = NodeMemory(n_nodes, hidden_d, n_layers)

    def get_weights(self):
        """Get model weights."""
        weights = super().get_weights()
        weights.append(self.memory_c.memory)
        return weights

    def set_weights(self, weights):
        """Set weights for the model.

        Args:
            weights: List[np.ndarray], value for new weights.
        """
        super().set_weights(weights)
        self.memory_c.memory = weights[3]

    def reset_memory_state(self):
        """Reset the node memory for hidden states."""
        self.memory_h.reset_state()
        self.memory_c.reset_state()

    def call(self, in_state: Tuple[Tensor, Tensor], training=False) -> Tensor:
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids_in = in_state
        ids, ids_id = _process_target_ids(ids_in)
        node_embs = tf.gather(node_embs, ids)
        out_sequential = tf.reshape(node_embs, [-1, 1, self.input_d])
        h = [tf.cast(to_tensor(h_tensor), tf.float32) for h_tensor in self.memory_h.get_memory(ids)]
        c = [tf.cast(to_tensor(c_tensor), tf.float32) for c_tensor in self.memory_c.get_memory(ids)]
        hc = [(h[i], c[i]) for i in range(self.n_layers)]
        out_result = self.base_model(out_sequential, initial_state=hc, training=training)
        out_sequential = out_result[0]
        new_hc = out_result[1:]
        new_h = [i[0] for i in new_hc]
        new_c = [i[1] for i in new_hc]
        self.memory_h.update_memory(tf.transpose(tf.convert_to_tensor(new_h), [1, 0, 2]), ids)
        self.memory_c.update_memory(tf.transpose(tf.convert_to_tensor(new_c), [1, 0, 2]), ids)
        out = self.simple_decoder((tf.reshape(out_sequential, [-1, self.hidden_d]), ids_id), training=training)
        return out


class LSTM_N(RnnFamily, NodeTrackingFamily):
    """LSTM sequential decoder."""

    def __init__(
        self,
        input_d: int,
        hidden_d: int,
        n_nodes: int,
        n_layers: int,
        simple_decoder: SimpleDecoder,
        **kwargs,
    ):
        """Create a LSTM sequential decoder.

        Args:
            input_d: int, input dimension.
            hidden_d: int, number of hidden cells.
            n_nodes: int, number of nodes.
            n_layers: int, number of lstm layers.
            simple_decoder: an instance of SimpleDecoder.
        """
        RnnFamily.__init__(self, hidden_d, n_nodes, n_layers, simple_decoder)
        NodeTrackingFamily.__init__(self, n_nodes)
        self.input_d = input_d
        self.base_model = TfRnn(
            [LSTMCell(hidden_d, **kwargs) for i in range(n_layers)],
            return_state=True,
            return_sequences=True,
            **kwargs,
        )
        self.memory_c = NodeMemory(n_nodes, hidden_d, n_layers)

    def get_weights(self):
        """Get model weights."""
        weights = RnnFamily.get_weights(self)
        weights.append(NodeTrackingFamily.get_weights(self))
        weights.append(self.memory_c.memory)
        return weights

    def set_weights(self, weights):
        """Set weights for the model.

        Args:
            weights: List[np.ndarray], value for new weights.
        """
        RnnFamily.set_weights(self, weights)
        NodeTrackingFamily.set_weights(self, weights[3])
        self.memory_c.memory = weights[4]

    def reset_memory_state(self):
        """Reset the node memory for hidden states."""
        self.memory_h.reset_state()
        self.memory_c.reset_state()
        NodeTrackingFamily.reset_node_memory(self)

    def call(self, in_state: Tuple[Tensor, Tensor], training=False) -> Tensor:
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids_in = in_state
        ids, ids_id = _process_target_ids(ids_in)
        node_embs = tf.gather(node_embs, ids)
        old_mem = tf.reshape(tf.cast(to_tensor(self.memory_node.get_memory(ids)), tf.float32), (node_embs.shape[0], 1))
        out_sequential = tf.reshape(node_embs, [-1, 1, self.input_d])
        h = [tf.cast(to_tensor(h_tensor), tf.float32) for h_tensor in self.memory_h.get_memory(ids)]
        c = [tf.cast(to_tensor(c_tensor), tf.float32) for c_tensor in self.memory_c.get_memory(ids)]
        hc = [(h[i], c[i]) for i in range(self.n_layers)]
        out_result = self.base_model(out_sequential, initial_state=hc, training=training)
        out_sequential = out_result[0]
        new_hc = out_result[1:]
        new_h = [i[0] for i in new_hc]
        new_c = [i[1] for i in new_hc]
        self.memory_h.update_memory(tf.transpose(tf.convert_to_tensor(new_h), [1, 0, 2]), ids)
        self.memory_c.update_memory(tf.transpose(tf.convert_to_tensor(new_c), [1, 0, 2]), ids)
        time_mem = self.simple_decoder((tf.reshape(out_sequential, [-1, self.hidden_d]), ids_id), training=training)
        new_mem = self.memory_layer(tf.concat((old_mem, node_embs), axis=1))  # N 1
        self.memory_node.update_memory(tf.reshape(new_mem, (node_embs.shape[0], 1, 1)), ids)
        combine_mem = tf.concat((time_mem, new_mem), axis=1)  # N 2
        out = self.output_layer(combine_mem)
        return out


class GRU(RnnFamily):
    """GRU sequential decoder."""

    def __init__(
        self,
        input_d: int,
        hidden_d: int,
        n_nodes: int,
        n_layers: int,
        simple_decoder: SimpleDecoder,
        memory_on_cpu: bool = False,
        **kwargs,
    ):
        """Create a LSTM sequential decoder.

        Args:
            input_d: int, input dimension.
            hidden_d: int, number of hidden cells.
            n_nodes: int, number of nodes.
            n_layers: int, number of GRU layers.
            simple_decoder: an instance of SimpleDecoder.
            memory_on_cpu: bool, whether to store hidden state memory on RAM.
        """
        super().__init__(hidden_d, n_nodes, n_layers, simple_decoder)
        self.input_d = input_d
        self.base_model = TfRnn([GRUCell(hidden_d) for i in range(n_layers)], return_state=True, **kwargs)


class RNN(RnnFamily):
    """RNN sequential decoder."""

    def __init__(
        self,
        input_d: int,
        hidden_d: int,
        n_nodes: int,
        n_layers: int,
        simple_decoder: SimpleDecoder,
        **kwargs,
    ):
        """Create a LSTM sequential decoder.

        Args:
            input_d: int, input dimension.
            hidden_d: int, number of hidden cells.
            n_nodes: int, number of nodes.
            n_layers: int, number of GRU layers.
            simple_decoder: an instance of SimpleDecoder.
        """
        super().__init__(hidden_d, n_nodes, n_layers, simple_decoder)
        self.input_d = input_d
        self.base_model = TfRnn(
            [SimpleRNNCell(hidden_d, **kwargs) for i in range(n_layers)],
            return_state=True,
            return_sequences=True,
            **kwargs,
        )


class SelfAttention(SlidingWindowFamily):
    """Temporal self-attention (SA) decoder."""

    def __init__(
        self,
        num_heads: int,
        input_d: int,
        embed_dims: List[int],
        n_nodes: int,
        window_size: int,
        simple_decoder: SimpleDecoder,
        **kwargs,
    ):
        """Create a sequential decoder.

        Args:
            num_heads: int, number of attention heads
            input_d: int, number of input feature
            embed_dims: List[int], list of hidden dimension for each SA layer.
            n_nodes: int number of nodes in the graph.
            window_size: int, the length of the sliding window
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        super().__init__(input_d, n_nodes, window_size, simple_decoder)
        self.nn_layers = nt_layers_list()
        for emb in embed_dims:
            self.nn_layers.append(TSA(out_dim=emb, num_heads=num_heads, **kwargs))

    def call(self, in_state: Tuple[Tensor, Tensor], training=False) -> Tensor:
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids_in = in_state
        ids, ids_id = _process_target_ids(ids_in)
        node_embs = tf.gather(node_embs, ids)
        self.memory.update_window(node_embs.numpy(), ids)
        input_windows = tf.convert_to_tensor(
            self.memory.get_memory(ids), dtype=tf.float32
        )  # [N, W, D] N serve as batch size in this case
        current = tf.identity(input_windows)
        for layer in self.nn_layers:
            new_current = layer(current, current, training=training)
            current = tf.identity(new_current)
        last_sequence = tf.slice(current, [0, self.window_size - 1, 0], [current.shape[0], 1, current.shape[2]])
        out = self.simple_decoder((tf.reshape(last_sequence, [-1, current.shape[2]]), ids_id), training=training)
        return out


class NaivePTSA(SlidingWindowFamily):
    """Positional Temporal self-attention (SA) decoder."""

    def __init__(
        self,
        num_heads: int,
        input_d: int,
        embed_dims: List[int],
        n_nodes: int,
        window_size: int,
        simple_decoder: SimpleDecoder,
        **kwargs,
    ):
        """Create a sequential decoder.

        Args:
            num_heads: int, number of attention heads
            input_d: int, number of input feature
            embed_dims: List[int], list of hidden dimension for each SA layer.
            n_nodes: int number of nodes in the graph.
            window_size: int, the length of the sliding window
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        super().__init__(input_d, n_nodes, window_size, simple_decoder)
        self.nn_layers = nt_layers_list()
        for emb in embed_dims:
            self.nn_layers.append(TSA(out_dim=emb, num_heads=num_heads, **kwargs))
        self.positional_embedding = tf.Variable(tf.random.uniform([window_size, input_d], dtype=tf.float32))

    def call(self, in_state: Tuple[Tensor, Tensor], training=False):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids_in = in_state
        ids, ids_id = _process_target_ids(ids_in)
        node_embs = tf.gather(node_embs, ids)
        self.memory.update_window(node_embs.numpy(), ids)
        input_windows = tf.convert_to_tensor(
            self.memory.get_memory(ids), dtype=tf.float32
        )  # N, W, D N serve as batch size in this case
        K = input_windows + self.positional_embedding
        for layer in self.nn_layers:
            K = layer(K, K, K, training=training)
        last_sequence = tf.slice(K, [0, self.window_size - 1, 0], [K.shape[0], 1, K.shape[2]])
        out = self.simple_decoder((tf.reshape(last_sequence, [-1, K.shape[2]]), ids_id), training=training)
        return out


class NodeTrackingPTSA(SlidingWindowFamily, NodeTrackingFamily):
    """Positional Temporal self-attention (SA) decoder."""

    def __init__(
        self,
        num_heads: int,
        input_d: int,
        embed_dims: List[int],
        n_nodes: int,
        window_size: int,
        simple_decoder: SimpleDecoder,
        **kwargs,
    ):
        """Create a sequential decoder.

        Args:
            num_heads: int, number of attention heads
            input_d: int, number of input feature
            embed_dims: List[int], list of hidden dimension for each SA layer.
            n_nodes: int number of nodes in the graph.
            window_size: int, the length of the sliding window
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        SlidingWindowFamily.__init__(self, input_d, n_nodes, window_size, simple_decoder)
        NodeTrackingFamily.__init__(self, n_nodes)
        self.nn_layers = nt_layers_list()
        self.output_dim = embed_dims[-1]
        for emb in embed_dims:
            self.nn_layers.append(TSA(out_dim=emb, num_heads=num_heads, **kwargs))
        self.positional_embedding = tf.Variable(tf.random.uniform([window_size, input_d], dtype=tf.float32))

    def reset_memory_state(self):
        """Reset all memory."""
        SlidingWindowFamily.reset_memory_state(self)
        NodeTrackingFamily.reset_node_memory(self)

    def call(self, in_state: Tuple[Tensor, Tensor], training=False):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids_in = in_state
        ids, ids_id = _process_target_ids(ids_in)
        node_embs = tf.gather(node_embs, ids)
        old_mem = tf.reshape(tf.cast(to_tensor(self.memory_node.get_memory(ids)), tf.float32), (node_embs.shape[0], 1))
        self.memory.update_window(node_embs.numpy(), ids)
        input_windows = tf.convert_to_tensor(
            self.memory.get_memory(ids), dtype=tf.float32
        )  # N, W, D N serve as batch size in this case
        K = input_windows + self.positional_embedding
        for layer in self.nn_layers:
            K = layer(K, K, K, training=training)
        last_sequence = tf.slice(K, [0, self.window_size - 1, 0], [K.shape[0], 1, K.shape[2]])
        time_mem = tf.reshape(last_sequence, [-1, K.shape[2]])  # N D
        time_mem = self.simple_decoder((time_mem, ids_id), training=training)  # N 1
        new_mem = self.memory_layer(tf.concat((old_mem, node_embs), axis=1))  # N 1
        self.memory_node.update_memory(tf.reshape(new_mem, (node_embs.shape[0], 1, 1)), ids)
        combine_mem = tf.concat((time_mem, new_mem), axis=1)  # N 2
        out = self.output_layer(combine_mem)
        # out = self.simple_decoder((combine_mem,ids_id),training=training)
        return out


def PTSA(
    num_heads: int,
    input_d: int,
    embed_dims: List[int],
    n_nodes: int,
    window_size: int,
    simple_decoder: SimpleDecoder,
    node_tracking: bool = False,
    **kwargs,
) -> SlidingWindowFamily:
    """Factory function to return the corresponding FTSA decoder."""
    model = None
    if node_tracking:
        model = NodeTrackingPTSA(num_heads, input_d, embed_dims, n_nodes, window_size, simple_decoder, **kwargs)
    else:
        model = NaivePTSA(num_heads, input_d, embed_dims, n_nodes, window_size, simple_decoder, **kwargs)

    return model


def FTSA(
    num_heads: int,
    input_d: int,
    embed_dims: List[int],
    n_nodes: int,
    window_size: int,
    time_kernel: int,
    time_agg: str,
    simple_decoder: SimpleDecoder,
    node_tracking: bool = False,
    **kwargs,
) -> SlidingWindowFamily:
    """Factory function to return the corresponding FTSA decoder."""
    model = None
    sum_ = "sum"
    concate_ = "concate"
    if time_agg == concate_:
        model = FTSAConcate(num_heads, input_d, embed_dims, n_nodes, window_size, time_kernel, simple_decoder, **kwargs)
    elif time_agg == sum_:
        if node_tracking:
            model = NodeTrackingFTSASum(
                num_heads, input_d, embed_dims, n_nodes, window_size, time_kernel, simple_decoder, **kwargs
            )
        else:
            model = FTSASum(num_heads, input_d, embed_dims, n_nodes, window_size, time_kernel, simple_decoder, **kwargs)
    else:
        raise ValueError(f"the input value of argument time_agg can only be {sum_} or {concate_}")
    return model


class FTSAConcate(SlidingWindowFamily):
    """Functional Temporal self-attention (SA) decoder with concatenate as time encoding aggregation method."""

    def __init__(
        self,
        num_heads: int,
        input_d: int,
        embed_dims: List[int],
        n_nodes: int,
        window_size: int,
        time_kernel: int,
        simple_decoder: SimpleDecoder,
        **kwargs,
    ):
        """Create a sequential decoder.

        Args:
            num_heads: int, number of attention heads
            input_d: int, number of input feature
            embed_dims: List[int], list of hidden dimension for each SA layer.
            n_nodes: int number of nodes in the graph.
            window_size: int, the length of the sliding window
            time_kernel: int, kernel size of the time2vec embedding
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        super().__init__(input_d, n_nodes, window_size, simple_decoder)
        self.nn_layers = nt_layers_list()
        for emb in embed_dims:
            self.nn_layers.append(TSA(out_dim=emb, num_heads=num_heads, **kwargs))
        self.time2vec = Time2Vec(time_kernel, 1, **kwargs)
        self.time_dimention = tf.reshape(tf.range(window_size, dtype=tf.float32), [1, -1])
        self.time_kernel = time_kernel
        self.input_d = input_d

    def build(self, input_shape):
        """Initiate model weigths."""
        self.wt = self.add_weight(
            shape=(self.input_d + self.time_kernel, self.input_d), initializer='uniform', trainable=True
        )
        super().build(input_shape)

    def call(self, in_state: Tuple[Tensor, Tensor], training=False):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids_in = in_state
        ids, ids_id = _process_target_ids(ids_in)
        node_embs = tf.gather(node_embs, ids)
        self.memory.update_window(node_embs.numpy(), ids)
        input_windows = tf.convert_to_tensor(
            self.memory.get_memory(ids), dtype=tf.float32
        )  # N, W, D N serve as batch size in this case
        time_encoding = tf.squeeze(self.time2vec(self.time_dimention))
        # input_with_time dimension: [N, W, D+T]
        input_with_time = tf.concat([input_windows, [time_encoding for i in range(node_embs.shape[0])]], -1)
        input_with_time = tf.matmul(input_with_time, self.wt)  # [N, W, D]
        for layer in self.nn_layers:
            input_with_time = layer(input_with_time, input_with_time, training=training)
        last_sequence = tf.slice(
            input_with_time, [0, self.window_size - 1, 0], [input_with_time.shape[0], 1, input_with_time.shape[2]]
        )
        out = self.simple_decoder(
            (tf.reshape(last_sequence, [-1, input_with_time.shape[2]]), ids_id), training=training
        )
        return out


class FTSASum(SlidingWindowFamily):
    """Functional Temporal self-attention (SA) decoder with summation as time encoding aggregation method."""

    def __init__(
        self,
        num_heads: int,
        input_d: int,
        embed_dims: List[int],
        n_nodes: int,
        window_size: int,
        time_kernel: int,
        simple_decoder: SimpleDecoder,
        **kwargs,
    ):
        """Create a sequential decoder.

        Args:
            num_heads: int, number of attention heads
            input_d: int, number of input feature
            embed_dims: List[int], list of hidden dimension for each SA layer.
            n_nodes: int number of nodes in the graph.
            window_size: int, the length of the sliding window
            time_kernel: int, kernel size of the time2vec embedding
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        super().__init__(input_d, n_nodes, window_size, simple_decoder)
        self.nn_layers = nt_layers_list()
        for emb in embed_dims:
            self.nn_layers.append(TSA(out_dim=emb, num_heads=num_heads, **kwargs))
        self.time2vec = Time2Vec(time_kernel, input_d, **kwargs)
        self.time_dimention = tf.reshape(tf.range(window_size, dtype=tf.float32), [1, -1])
        self.time_kernel = time_kernel
        self.feature_size = input_d

    def build(self, input_shape):
        """Initiate model weights."""
        self.wk = self.add_weight(shape=(self.time_kernel, 1), initializer='uniform', trainable=True)

    def call(self, in_state: Tuple[Tensor, Tensor], training=False):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids_in = in_state
        ids, ids_id = _process_target_ids(ids_in)
        node_embs = tf.gather(node_embs, ids)
        self.memory.update_window(node_embs.numpy(), ids)

        input_windows = tf.convert_to_tensor(
            self.memory.get_memory(ids), dtype=tf.float32
        )  # N, W, D N serve as batch size in this case
        time_encoding = self.time2vec(self.time_dimention)  # 1, W, D, K
        time_encoding = tf.reshape(time_encoding, [-1, self.feature_size, self.time_kernel])  # W D K
        input_with_time = tf.einsum('abc, bch->abch', input_windows, time_encoding)  # N W D K
        input_with_time = tf.transpose(input_with_time, [0, 1, 3, 2])  # N W k D
        input_with_time = tf.reshape(input_with_time, [-1, self.window_size * self.time_kernel, self.feature_size])
        # N W*K D
        for layer in self.nn_layers:
            input_with_time = layer(input_with_time, input_with_time, training=training)

        input_with_time = tf.reshape(input_with_time, [node_embs.shape[0], self.window_size, self.time_kernel, -1])
        input_with_time = tf.tensordot(input_with_time, self.wk, [[2], [0]])
        input_with_time = tf.reshape(input_with_time, [node_embs.shape[0], self.window_size, -1])
        last_sequence = tf.slice(
            input_with_time, [0, self.window_size - 1, 0], [input_with_time.shape[0], 1, input_with_time.shape[2]]
        )
        last_sequence = tf.reshape(last_sequence, [-1, input_with_time.shape[2]])
        out = self.simple_decoder((last_sequence, ids_id), training=training)
        return out


class NodeTrackingFTSAConcate(SlidingWindowFamily):
    """Functional Temporal self-attention (SA) decoder with concatenate as time encoding aggregation method."""

    def __init__(
        self,
        num_heads: int,
        input_d: int,
        embed_dims: List[int],
        n_nodes: int,
        window_size: int,
        time_kernel: int,
        simple_decoder: SimpleDecoder,
        **kwargs,
    ):
        """Create a sequential decoder.

        Args:
            num_heads: int, number of attention heads
            input_d: int, number of input feature
            embed_dims: List[int], list of hidden dimension for each SA layer.
            n_nodes: int number of nodes in the graph.
            window_size: int, the length of the sliding window
            time_kernel: int, kernel size of the time2vec embedding
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        super().__init__(input_d, n_nodes, window_size, simple_decoder)
        self.nn_layers = nt_layers_list()
        for emb in embed_dims:
            self.nn_layers.append(TSA(out_dim=emb, num_heads=num_heads, **kwargs))
        self.time2vec = Time2Vec(time_kernel, 1, **kwargs)
        self.time_dimention = tf.range(window_size, dtype=tf.float32)
        self.time_kernel = time_kernel
        self.input_d = input_d

    def build(self, input_shape):
        """Initiate model weigths."""
        self.wt = self.add_weight(
            shape=(self.input_d + self.time_kernel, self.input_d), initializer='uniform', trainable=True
        )
        super().build(input_shape)

    def reset_memory_state(self):
        """Reset all memory."""
        SlidingWindowFamily.reset_memory_state(self)
        NodeTrackingFamily.reset_node_memory(self)

    def call(self, in_state: Tuple[Tensor, Tensor], training=False):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids_in = in_state
        ids, ids_id = _process_target_ids(ids_in)
        node_embs = tf.gather(node_embs, ids)
        self.memory.update_window(node_embs.numpy(), ids)
        input_windows = tf.convert_to_tensor(
            self.memory.get_memory(ids), dtype=tf.float32
        )  # N, W, D N serve as batch size in this case
        time_encoding = tf.squeeze(self.time2vec(self.time_dimention))
        # input_with_time dimension: [N, W, D+T]
        input_with_time = tf.concat([input_windows, [time_encoding for i in range(node_embs.shape[0])]], -1)
        input_with_time = tf.matmul(input_with_time, self.wt)  # [N, W, D]
        for layer in self.nn_layers:
            input_with_time = layer(input_with_time, input_with_time, training=training)
        last_sequence = tf.slice(
            input_with_time, [0, self.window_size - 1, 0], [input_with_time.shape[0], 1, input_with_time.shape[2]]
        )
        out = self.simple_decoder(
            (tf.reshape(last_sequence, [-1, input_with_time.shape[2]]), ids_id), training=training
        )
        return out


class NodeTrackingFTSASum(SlidingWindowFamily, NodeTrackingFamily):
    """Functional Temporal self-attention (SA) decoder with summation as time encoding aggregation method."""

    def __init__(
        self,
        num_heads: int,
        input_d: int,
        embed_dims: List[int],
        n_nodes: int,
        window_size: int,
        time_kernel: int,
        simple_decoder: SimpleDecoder,
        **kwargs,
    ):
        """Create a sequential decoder.

        Args:
            num_heads: int, number of attention heads
            input_d: int, number of input feature
            embed_dims: List[int], list of hidden dimension for each SA layer.
            n_nodes: int number of nodes in the graph.
            window_size: int, the length of the sliding window
            time_kernel: int, kernel size of the time2vec embedding
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        SlidingWindowFamily.__init__(self, input_d, n_nodes, window_size, simple_decoder)
        NodeTrackingFamily.__init__(self, n_nodes)
        self.nn_layers = nt_layers_list()
        self.output_dim = embed_dims[-1]
        for emb in embed_dims:
            self.nn_layers.append(TSA(out_dim=emb, num_heads=num_heads, **kwargs))
        self.time2vec = Time2Vec(time_kernel, input_d, **kwargs)
        self.time_dimention = tf.reshape(tf.range(window_size, dtype=tf.float32), [1, -1])
        self.time_kernel = time_kernel
        self.feature_size = input_d

    def build(self, input_shape):
        """Initiate model weights."""
        self.wk = self.add_weight(shape=(self.time_kernel, 1), initializer='uniform', trainable=True)

    def reset_memory_state(self):
        """Reset all memory."""
        SlidingWindowFamily.reset_memory_state(self)
        NodeTrackingFamily.reset_node_memory(self)

    def call(self, in_state: Tuple[Tensor, Tensor], training=False):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids_in = in_state
        ids, ids_id = _process_target_ids(ids_in)
        node_embs = tf.gather(node_embs, ids)
        old_mem = tf.reshape(tf.cast(to_tensor(self.memory_node.get_memory(ids)), tf.float32), (node_embs.shape[0], 1))
        self.memory.update_window(node_embs.numpy(), ids)

        input_windows = tf.convert_to_tensor(
            self.memory.get_memory(ids), dtype=tf.float32
        )  # N, W, D N serve as batch size in this case
        time_encoding = self.time2vec(self.time_dimention)  # 1, W, D, K
        time_encoding = tf.reshape(time_encoding, [-1, self.feature_size, self.time_kernel])  # W D K
        input_with_time = tf.einsum('abc, bch->abch', input_windows, time_encoding)  # N W D K
        input_with_time = tf.transpose(input_with_time, [0, 1, 3, 2])  # N W k D
        input_with_time = tf.reshape(input_with_time, [-1, self.window_size * self.time_kernel, self.feature_size])
        # N W*K D
        for layer in self.nn_layers:
            input_with_time = layer(input_with_time, input_with_time, training=training)
        input_with_time = tf.reshape(input_with_time, [node_embs.shape[0], self.window_size, self.time_kernel, -1])
        input_with_time = tf.tensordot(input_with_time, self.wk, [[2], [0]])
        input_with_time = tf.reshape(input_with_time, [node_embs.shape[0], self.window_size, -1])
        last_sequence = tf.slice(
            input_with_time, [0, self.window_size - 1, 0], [input_with_time.shape[0], 1, input_with_time.shape[2]]
        )
        new_mem = self.memory_layer(tf.concat((old_mem, node_embs), axis=1))  # N 1
        self.memory_node.update_memory(tf.reshape(new_mem, (node_embs.shape[0], 1, 1)), ids)
        time_mem = tf.reshape(last_sequence, [-1, self.output_dim])  # N D
        time_mem = self.simple_decoder((time_mem, ids_id), training=training)  # N 1
        combine_mem = tf.concat((time_mem, new_mem), axis=1)  # N 2
        out = self.output_layer(combine_mem)
        return out


class Conv1D(SlidingWindowFamily):
    """1-D convolutional network decoder."""

    def __init__(
        self,
        input_d: int,
        embed_dims: List[int],
        n_nodes: int,
        window_size: int,
        simple_decoder: SimpleDecoder,
        **kwargs,
    ):
        """Create a sequential decoder.

        Args:
            num_heads: int, number of attention heads
            input_d: int, number of input feature
            embed_dims: List[int], list of hidden dimension for each SA layer.
            n_nodes: int number of nodes in the graph.
            window_size: int, the length of the sliding window
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        super().__init__(input_d, n_nodes, window_size, simple_decoder)
        self.nn_layers = nt_layers_list()
        self.nn_layers.append(Conv1d(input_d, embed_dims[0], window_size, **kwargs))
        for i in range(1, len(embed_dims)):
            self.nn_layers.append(Conv1d(embed_dims[i - 1], embed_dims[i], window_size, **kwargs))

    def call(self, in_state: Tuple[Tensor, Tensor], training=False):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids_in = in_state
        ids, ids_id = _process_target_ids(ids_in)
        node_embs = tf.gather(node_embs, ids)
        self.memory.update_window(node_embs.numpy(), ids)
        input_windows = tf.convert_to_tensor(
            self.memory.get_memory(ids), dtype=tf.float32
        )  # [N, W, D] N serve as batch size in this case
        current = tf.identity(input_windows)
        for layer in self.nn_layers:
            new_current = layer(current, training=training)
            current = tf.identity(new_current)
        last_sequence = tf.slice(current, [0, self.window_size - 1, 0], [current.shape[0], 1, current.shape[2]])
        out = self.simple_decoder((tf.reshape(last_sequence, [-1, current.shape[2]]), ids_id), training=training)
        return out


class Conv1D_N(SlidingWindowFamily, NodeTrackingFamily):
    """1-D convolutional network decoder with Node memory layer."""

    def __init__(
        self,
        input_d: int,
        embed_dims: List[int],
        n_nodes: int,
        window_size: int,
        simple_decoder: SimpleDecoder,
        **kwargs,
    ):
        """Create a sequential decoder.

        Args:
            num_heads: int, number of attention heads
            input_d: int, number of input feature
            embed_dims: List[int], list of hidden dimension for each SA layer.
            n_nodes: int number of nodes in the graph.
            window_size: int, the length of the sliding window
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        SlidingWindowFamily.__init__(self, input_d, n_nodes, window_size, simple_decoder)
        NodeTrackingFamily.__init__(self, n_nodes)
        self.nn_layers = nt_layers_list()
        self.nn_layers.append(Conv1d(input_d, embed_dims[0], window_size, **kwargs))
        for i in range(1, len(embed_dims)):
            self.nn_layers.append(Conv1d(embed_dims[i - 1], embed_dims[i], window_size, **kwargs))

    def reset_memory_state(self):
        """Reset all memory."""
        SlidingWindowFamily.reset_memory_state(self)
        NodeTrackingFamily.reset_node_memory(self)

    def call(self, in_state: Tuple[Tensor, Tensor], training=False):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids_in = in_state
        ids, ids_id = _process_target_ids(ids_in)
        node_embs = tf.gather(node_embs, ids)
        self.memory.update_window(node_embs.numpy(), ids)
        old_mem = tf.reshape(tf.cast(to_tensor(self.memory_node.get_memory(ids)), tf.float32), (node_embs.shape[0], 1))
        input_windows = tf.convert_to_tensor(
            self.memory.get_memory(ids), dtype=tf.float32
        )  # [N, W, D] N serve as batch size in this case
        current = tf.identity(input_windows)
        for layer in self.nn_layers:
            new_current = layer(current, training=training)
            current = tf.identity(new_current)
        last_sequence = tf.slice(current, [0, self.window_size - 1, 0], [current.shape[0], 1, current.shape[2]])
        time_mem = self.simple_decoder((tf.reshape(last_sequence, [-1, current.shape[2]]), ids_id), training=training)
        new_mem = self.memory_layer(tf.concat((old_mem, node_embs), axis=1))  # N 1
        self.memory_node.update_memory(tf.reshape(new_mem, (node_embs.shape[0], 1, 1)), ids)
        combine_mem = tf.concat((time_mem, new_mem), axis=1)  # N 2
        out = self.output_layer(combine_mem)
        return out
