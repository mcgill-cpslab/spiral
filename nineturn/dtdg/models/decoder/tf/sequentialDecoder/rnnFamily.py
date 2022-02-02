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
from datetime import datetime
import copy
from abc import abstractmethod
from typing import List, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras, Tensor
from tensorflow.keras import layers

from nineturn.dtdg.models.decoder.torch.simpleDecoder import SimpleDecoder
from nineturn.core.commonF import to_tensor


class NodeMemory:
    """NodeMemory to remember states for each node."""

    def __init__(
        self, n_nodes: int, hidden_d: int, n_layers: int):
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
        #self.memory =tf.Variable(np.random.rand( self.n_nodes, self.n_layers, self.hidden_d),dtype= tf.float32)
        self.memory = np.random.rand( self.n_nodes, self.n_layers, self.hidden_d)

    def update_memory(self, new_memory, inx):
        """Update memory."""
        #for i in range(len(inx)):
        #    self.memory[inx[i]].assign(new_memory[i])
        self.memory[inx.numpy()] = new_memory.numpy()

    def get_memory(self, inx):
        """Retrieve node memory by index."""
        #return tf.gather(self.memory, inx)
        return to_tensor(self.memory[inx.numpy()])


class SequentialDecoder(layers.Layer):
    """Prototype of sequential decoders."""

    def __init__(self, hidden_d: int, n_nodes: int,n_layers:int, simple_decoder: SimpleDecoder):
        """Create a sequential decoder.

        Args:
            hidden_d: int, the hidden state's dimension.
            n_nodes: int, number of nodes to remember.
            simple_decoder: SimpleDecoder, the outputing simple decoder.
            device: str or torch.device, the device this model will run. mainly for node memory.
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_d = hidden_d
        self.n_layers = n_layers
        self.mini_batch = False
        self.base_model = None
        self.simple_decoder = simple_decoder
        self.memory_h = NodeMemory(n_nodes, hidden_d, n_layers)

    def set_mini_batch(self, mini_batch: bool = True):
        """Set to batch training mode."""
        self.mini_batch = mini_batch


    def reset_memory_state(self):
        """Reset the node memory for hidden states."""
        self.memory_h.reset_state()

    def call(self, in_state: Tuple[Tensor, List[int]]):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids = in_state
        if not self.mini_batch:
            node_embs = tf.gather(node_embs,ids)
        out_sequential = tf.reshape(node_embs, [-1, 1, self.input_d])
        h = tf.transpose(self.memory_h.get_memory(ids), [1,0,2])
        t1 = datetime.now()
        out_result= self.base_model(out_sequential)
        t2 = datetime.now()
        out_sequential = out_result[0]
        new_h = out_result[1:]
        self.memory_h.update_memory(tf.transpose(tf.squeeze(tf.convert_to_tensor(new_h)),[1,0,2]), ids)
        t3 = datetime.now()
        out = self.simple_decoder((tf.reshape(out_sequential, [-1, self.hidden_d]), ids))
        t4 = datetime.now()
        return out


class LSTM(SequentialDecoder):
    """LSTM sequential decoder."""

    def __init__(
        self,
        input_d: int,
        hidden_d: int,
        n_nodes: int,
        n_layers: int,
        simple_decoder: SimpleDecoder,
        memory_on_cpu:bool=False,
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
        self.base_model = self.base_model = [layers.LSTM(hidden_d, return_state=True, **kwargs) for i in range(n_layers)]
        self.memory_c = NodeMemory(n_nodes, hidden_d, n_layers)

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
        if not self.mini_batch:
            node_embs = node_embs[ids]
        out_sequential = node_embs.reshape(-1, 1, self.input_d)
        h = self.memory_h.get_memory(ids)
        c = self.memory_c.get_memory(ids)
        new_h = []
        new_c = []
        for i in range(self.n_layers):
            out_sequential, h_temp,c_temp = self.base_model[i](out_sequential, h[i])
            new_h.append(h_temp)
            new_c.append(c_temp)
        self.memory_h.update_memory(tf.Tensor(new_h), ids)
        self.memory_c.update_memory(tf.Tensor(new_c), ids)
        out = self.simple_decoder((out_sequential.reshape(-1, self.hidden_d), ids))
        return out


class GRU(SequentialDecoder):
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
        self.base_model = layers.RNN([layers.GRUCell(ahidden_d) for i in range(n_layers)], return_state = True,
                **kwargs)



class RNN(SequentialDecoder):
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
        self.base_model = layers.RNN(
                [layers.SimpleRNNCell(hidden_d, **kwargs) for i in range(n_layers)],
                return_state=True,return_sequences=True,
                **kwargs
                )
