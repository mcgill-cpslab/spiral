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

import numpy as np
import tensorflow as tf
from numpy import ndarray as Ndarray
from tensorflow import Tensor
from tensorflow.keras import layers

from spiro.core.commonF import to_tensor
from spiro.core.errors import DimensionError
from spiro.core.logger import get_logger
from spiro.dtdg.models.decoder.tf.simpleDecoder import SimpleDecoder

logger = get_logger()


def _process_target_ids(ids_in: Tensor) -> Tuple[Ndarray, Tensor]:
    ids_rank = tf.rank(ids_in).numpy()
    if ids_rank == 1:
        ids = ids_in.numpy()
        ids_id = ids_in
    elif ids_rank == 2:
        ids, ids_id = tf.unique(tf.reshape(ids_in, [-1]))
        ids = ids.numpy()
        ids_id = tf.reshape(ids_id, [-1, 2])  # map original node index to new index in the selected node embeding
    else:
        message = f"""The index to predict in the input must be of rank 1 for node prediction or rank 2 for edge
        prediction. But get an input index of rank {ids_rank}"""
        logger.error(message)
        raise DimensionError(message)
    return (ids, ids_id)


class NodeMemory:
    """NodeMemory to remember states for each node."""

    def __init__(self, n_nodes: int, hidden_d: int, n_layers: int):
        """Create a node memory based on the number of nodes and the state dimension.

        Args:
            n_nodes: int, number of nodes to remember.
            hidden_d: int, the hidden state's dimension.
            n_layers: int, number of targeting rnn layers.
        """
        self.n_nodes = n_nodes
        self.hidden_d = hidden_d
        self.n_layers = n_layers
        self.reset_state()

    def reset_state(self):
        """Reset the memory to a random tensor."""
        self.memory = np.zeros((self.n_nodes, self.n_layers, self.hidden_d))

    def update_memory(self, new_memory: Tensor, inx: List[int]):
        """Update memory [N,L,D]."""
        self.memory[inx] = new_memory.numpy()

    def get_memory(self, inx: List[int]) -> Ndarray:
        """Retrieve node memory by index.Return shape [L,N,D]."""
        selected = self.memory[inx]
        result = np.einsum('lij->ilj', selected)
        return result


class BaseModel(layers.Layer):
    """Prototype of sliding window based sequential decoders."""

    def __init__(self, hidden_d: int, simple_decoder: SimpleDecoder):
        """Create a sequential decoder.

        Args:
            hidden_d: int, the hidden state's dimension.
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        super().__init__()
        self.hidden_d = hidden_d
        self.mini_batch = False
        self.base_model = None
        self.simple_decoder = simple_decoder
        self.training_mode = True

    def training(self):
        """Set training mode to be true."""
        self.training_mode = True

    def eval_mode(self):
        """Set training mode to false."""
        self.training_mode = False

    def set_mini_batch(self, mini_batch: bool = True):
        """Set to batch training mode."""
        self.mini_batch = mini_batch


class RnnFamily(BaseModel):
    """Prototype of RNN family sequential decoders."""

    def __init__(self, hidden_d: int, n_nodes: int, n_layers: int, simple_decoder: SimpleDecoder):
        """Create a sequential decoder.

        Args:
            hidden_d: int, the hidden state's dimension.
            n_nodes: int, number of nodes to remember.
            n_layers: int, number of RNN layers.
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        super().__init__(hidden_d, simple_decoder)
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.memory_h = NodeMemory(n_nodes, hidden_d, n_layers)

    def reset_memory_state(self):
        """Reset the node memory for hidden states."""
        self.memory_h.reset_state()

    def get_weights(self) -> List[Ndarray]:
        """Get model weights.

        Return:
            a list of model weights
        """
        return [self.base_model.get_weights(), self.simple_decoder.get_weights(), self.memory_h.memory]

    def set_weights(self, weights: List[Ndarray]):
        """Set weights for the model.

        Args:
            weights: List[Ndarray], value for new weights.
        """
        self.base_model.set_weights(weights[0])
        self.simple_decoder.set_weights(weights[1])
        self.memory_h.memory = weights[2]

    def call(self, in_state: Tuple[Tensor, Tensor], training: bool = False) -> Tensor:
        """Forward function.

        Args:
            in_state: the input from encoder.
            training: boolean, training mode or not

        Return:
            the prediction
        """
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids_in = in_state
        ids, ids_id = _process_target_ids(ids_in)
        node_embs = tf.gather(node_embs, ids)
        out_sequential = tf.reshape(node_embs, [-1, 1, self.input_d])
        h = [tf.cast(to_tensor(h_tensor), tf.float32) for h_tensor in self.memory_h.get_memory(ids)]
        out_result = self.base_model(out_sequential, initial_state=h, training=training)
        out_sequential = out_result[0]
        new_h = out_result[1:]
        self.memory_h.update_memory(tf.transpose(tf.convert_to_tensor(new_h), [1, 0, 2]), ids)
        out = self.simple_decoder((tf.reshape(out_sequential, [-1, self.hidden_d]), ids_id), training=training)
        return out


class SlidingWindow:
    """SlidingWindow."""

    def __init__(self, n_nodes: int, input_d: int, window_size: int):
        """Create a node memory based on the number of nodes and the state dimension.

        Args:
            n_nodes: int, number of nodes to remember.
            input_d: int, the number of input features..
            window_size: int, number of snapshots in the sliding window..
        """
        self.n_nodes = n_nodes
        self.window_size = window_size
        self.input_d = input_d
        self.reset_state()

    def reset_state(self):
        """Reset the memory to a random tensor."""
        self.memory = np.zeros((self.n_nodes, self.window_size, self.input_d))

    def update_window(self, new_window: Ndarray, inx: List[int]):
        """Update memory with input window.

        Args:
            new_window: numpy array,
            inx: the node for which  to update memory.
        """
        self.memory[inx, :-1, :] = self.memory[inx, 1:, :]
        self.memory[inx, -1, :] = new_window

    def get_memory(self, inx: List[int]) -> Ndarray:
        """Retrieve node memory by index.

        Args:
            inx: the nodes to retrieve the memory
        """
        return self.memory[inx]


class SlidingWindowFamily(BaseModel):
    """Prototype of sliding window based sequential decoders."""

    def __init__(self, input_d: int, n_nodes: int, window_size: int, simple_decoder: SimpleDecoder):
        """Create a sequential decoder.

        Args:
            input_d: int, the hidden state's dimension.
            n_nodes: int, total number of nodes.
            window_size: int, the length of the sliding window
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        super().__init__(input_d, simple_decoder)
        self.window_size = window_size
        self.training_mode = True
        self.n_nodes = n_nodes
        self.memory = SlidingWindow(self.n_nodes, self.hidden_d, self.window_size)

    def reset_memory_state(self):
        """Reset the model's memory of the sliding window."""
        self.memory.reset_state()


class NodeTrackingFamily:
    """Prototype of sliding window based sequential decoders."""

    def __init__(self, n_nodes: int):
        """Create a sequential decoder.

        Args:
            input_d: int, the hidden state's dimension.
            n_nodes: int, total number of nodes.
            window_size: int, the length of the sliding window
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        self.memory_node = NodeMemory(n_nodes, 1, 1)
        self.memory_layer = tf.keras.layers.Dense(1)
        self.output_layer = tf.keras.layers.Dense(1)

    def reset_node_memory(self):
        """Reset the node memory."""
        self.memory_node.reset_state()

    def get_weights(self) -> List[Ndarray]:
        """Return the value of trainable weights."""
        return [self.memory_layer.get_weights(), self.output_layer.get_weights(), self.memory_node.memory]

    def set_weights(self, weights: List[Ndarray]):
        """Set the trainable weights value to the input ones."""
        self.memory_layer.set_weights(weights[0])
        self.output_layer.set_weights(weights[1])
        self.memory_node.memory = weights[2]
