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
"""Pytorch based sequential decoder. Designed specially for dynamic graph learning."""
import copy
from abc import abstractmethod
from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import MultiHeadAttention, RNN as TfRnn, LSTMCell, GRUCell, SimpleRNNCell

from nineturn.core.commonF import to_tensor
from nineturn.dtdg.models.decoder.tf.simpleDecoder import SimpleDecoder
from nineturn.dtdg.models.decoder.tf.sequentialDecoder.baseModel import BaseModel

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
        self.memory = np.random.rand(self.n_nodes, self.n_layers, self.hidden_d)

    def update_memory(self, new_memory, inx):
        """Update memory [N,L,D]."""
        self.memory[inx] = new_memory.numpy()

    def get_memory(self, inx):
        """Retrieve node memory by index.Return shape [L,N,D]."""
        selected = self.memory[inx]
        result = np.einsum('lij->ilj', selected)
        return result



class RnnFamily(BaseModel):
    """Prototype of RNN family sequential decoders."""

    def __init__(self, hidden_d: int, n_nodes: int, n_layers: int, simple_decoder: SimpleDecoder):
        """Create a sequential decoder.

        Args:
            hidden_d: int, the hidden state's dimension.
            n_nodes: int, number of nodes to remember.
            simple_decoder: SimpleDecoder, the outputing simple decoder.
            device: str or torch.device, the device this model will run. mainly for node memory.
        """
        super().__init__(hidden_d, simple_decoder)
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.memory_h = NodeMemory(n_nodes, hidden_d, n_layers)

    def reset_memory_state(self):
        """Reset the node memory for hidden states."""
        self.memory_h.reset_state()

    def get_weights(self):
        return [self.base_model.get_weights(), self.simple_decoder.get_weights(), self.memory_h.memory]

    def set_weights(self, weights):
        super().set_weights(weights)
        self.memory_h.memory = weights[2]
    
    def call(self, in_state: Tuple[Tensor, List[int]]):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids = in_state
        ids = ids.numpy()
        node_embs = tf.gather(node_embs, ids)
        out_sequential = tf.reshape(node_embs, [-1, 1, self.input_d])
        h = [tf.cast(to_tensor(h_tensor), tf.float32) for h_tensor in self.memory_h.get_memory(ids)]
        out_result = self.base_model(out_sequential, initial_state=h)
        out_sequential = out_result[0]
        if self.training_mode:
            new_h = out_result[1:]
            self.memory_h.update_memory(tf.transpose(tf.squeeze(tf.convert_to_tensor(new_h)), [1, 0, 2]), ids)
        out = self.simple_decoder((tf.reshape(out_sequential, [-1, self.hidden_d]), ids))
        return out


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
            memory_on_cpu: bool, always put node memory to cpu.
        """
        super().__init__(hidden_d, n_nodes, n_layers, simple_decoder)
        self.input_d = input_d
        self.base_model = TfRnn(
            [LSTMCell(hidden_d, **kwargs) for i in range(n_layers)],
            return_state=True,
            return_sequences=True,
            **kwargs,
        )
        self.memory_c = NodeMemory(n_nodes, hidden_d, n_layers)

    def get_weights(self):
        weights = super().get_weights()
        weights.append(self.memory_c.memory)
        return weights

    def set_weights(self, weights):
        super().set_weights(weights)
        self.memory_c.memory = weights[3]

    def reset_memory_state(self):
        """Reset the node memory for hidden states."""
        self.memory_h.reset_state()
        self.memory_c.reset_state()

    def call(self, in_state: Tuple[Tensor, List[int]]):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids = in_state
        ids = ids.numpy()
        node_embs = tf.gather(node_embs, ids)
        out_sequential = tf.reshape(node_embs, [-1, 1, self.input_d])
        h = [tf.cast(to_tensor(h_tensor), tf.float32) for h_tensor in self.memory_h.get_memory(ids)]
        c = [tf.cast(to_tensor(c_tensor), tf.float32) for c_tensor in self.memory_c.get_memory(ids)]
        hc = [(h[i], c[i]) for i in range(self.n_layers)]
        out_result = self.base_model(out_sequential, initial_state=hc)
        out_sequential = out_result[0]
        if self.training_mode:
            new_hc = out_result[1:]
            new_h = [i[0] for i in new_hc]
            new_c = [i[1] for i in new_hc]
            self.memory_h.update_memory(tf.transpose(tf.squeeze(tf.convert_to_tensor(new_h)), [1, 0, 2]), ids)
            self.memory_c.update_memory(tf.transpose(tf.squeeze(tf.convert_to_tensor(new_c)), [1, 0, 2]), ids)
        out = self.simple_decoder((tf.reshape(out_sequential, [-1, self.hidden_d]), ids))
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



class SlidingWindow:
    """SlidingWindow."""

    def __init__(self, n_nodes: int, input_d: int, window_size: int):
        """Create a node memory based on the number of nodes and the state dimension.

        Args:
            n_nodes: int, number of nodes to remember.
            hidden_d: int, the hidden state's dimension.
            n_layers: int, number of targeting rnn layers.
        """
        self.n_nodes = n_nodes
        self.window_size = window_size
        self.input_d = input_d
        self.reset_state()

    def reset_state(self):
        """Reset the memory to a random tensor."""
        self.memory = [[[0 for d in self.input_d] for w in self.window_size] for n in self.n_nodes]

    def update_window(self, new_window, inx):
        """Update memory [N,W,D]."""
        new_memory = self.memory[inx]

        self.memory[inx] = new_memory.numpy()

    def get_memory(self, inx):
        """Retrieve node memory by index.Return shape [L,N,D]."""
        selected = self.memory[inx]
        result = np.einsum('lij->ilj', selected)
        return result


"""
class SlidingWindowFamily(BaseModel):
    """Prototype of sliding window based sequential decoders."""

    def __init__(self, hidden_d: int, window_size: int,  simple_decoder: SimpleDecoder):
        """Create a sequential decoder.

        Args:
            hidden_d: int, the hidden state's dimension.
            window_size: int, the length of the sliding window
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        super().__init__(hidden_d, simple_decoder)
        self.window_size = window_size
        self.training_mode = True

    

class SelfAttention(SlidingWindowFamily):
    
    def __init__(self,num_heads:int, key_dim: int, hidden_d: int, window_size: int,  simple_decoder: SimpleDecoder, **kwargs):
        """Create a sequential decoder.

        Args:
            num_heads: int, number of attention heads
            key_dim: int, dimension of input key
            hidden_d: int, the hidden state's dimension.
            window_size: int, the length of the sliding window
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        super().__init__(hidden_d, simple_decoder)
        self.base_model = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim = hidden_d, **kwargs)
    def call(self, in_state: Tuple[Tensor, List[int]]):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids = in_state
        ids = ids.numpy()
        node_embs = tf.gather(node_embs, ids)
        out_sequential = tf.reshape(node_embs, [-1, 1, self.input_d])
        h = [tf.cast(to_tensor(h_tensor), tf.float32) for h_tensor in self.memory_h.get_memory(ids)]
        out_result = self.base_model(out_sequential, initial_state=h)
        out_sequential = out_result[0]
        if self.training_mode:
            new_h = out_result[1:]
            self.memory_h.update_memory(tf.transpose(tf.squeeze(tf.convert_to_tensor(new_h)), [1, 0, 2]), ids)
        out = self.simple_decoder((tf.reshape(out_sequential, [-1, self.hidden_d]), ids))
        return out
"""
