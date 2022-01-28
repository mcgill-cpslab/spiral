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

from abc import abstractmethod
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from nineturn.dtdg.models.decoder.torch.simpleDecoder import SimpleDecoder


class NodeMemory:
    """NodeMemory to remember states for each node."""

    def __init__(
        self, n_nodes: int, hidden_d: int, n_layers: int, device: Union[str, torch.device], memory_on_cpu: bool = False
    ):
        """Create a node memory based on the number of nodes and the state dimension.

        Args:
            n_nodes: int, number of nodes to remember.
            hidden_d: int, the hidden state's dimension.
            n_layers: int, number of targeting rnn layers.
            device: str or torch.device, the device this model will run. mainly for node memory.
            memory_on_cpu: bool, whether to store memory in ram instead of the running device.
        """
        self.n_nodes = n_nodes
        self.hidden_d = hidden_d
        self.n_layers = n_layers
        self.device = device
        if memory_on_cpu:
            self.device = "cpu"
        self.reset_state()

    def reset_state(self):
        """Reset the memory to a random tensor."""
        self.memory = torch.randn(self.n_layers, self.n_nodes, self.hidden_d).to(self.device)

    def update_memory(self, new_memory, inx):
        """Update memory."""
        self.memory[:, inx, :] = new_memory.to(self.device)

    def get_memory(self, inx, device):
        """Retrieve node memory by index."""
        return self.memory[:, inx, :]


class SequentialDecoder(nn.Module):
    """Prototype of sequential decoders."""

    def __init__(self, hidden_d: int, n_nodes: int, simple_decoder: SimpleDecoder, device: Union[str, torch.device]):
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
        self.mini_batch = False
        self.simple_decoder = simple_decoder
        self.device = device

    def set_mini_batch(self, mini_batch: bool = True):
        """Set to batch training mode."""
        self.mini_batch = mini_batch

    @abstractmethod
    def reset_memory_state(self, new_memory):
        """Reset the node memory for hidden states."""
        pass

    @abstractmethod
    def forward(self, in_sample: Tuple[torch.Tensor, List]) -> torch.Tensor:
        """All SequentialDecoder subclass should have a forward function.

        Args:
            in_sample: tuple, first entry is a node-wise embedding from an encoder,
                       second entry is the list[int] of targeted node ids.

        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        pass


class LSTM(SequentialDecoder):
    """LSTM sequential decoder."""

    def __init__(
        self,
        input_d: int,
        hidden_d: int,
        n_nodes: int,
        n_layers: int,
        simple_decoder: SimpleDecoder,
        device: Union[str, torch.device],
        memory_on_cpu: bool = False,
        **kwargs,
    ):
        """Create a LSTM sequential decoder.

        Args:
            input_d: int, input dimension.
            hidden_d: int, number of hidden cells.
            n_nodes: int, number of nodes.
            n_layers: int, number of lstm layers.
            simple_decoder: an instance of SimpleDecoder.
            device: str or torch.device, the device this model will run. Mainly for node memory.
            memory_on_cpu: bool, whether to store memory in RAM instead of the running device.
        """
        super().__init__(hidden_d, n_nodes, simple_decoder, device)
        self.input_d = input_d
        self.lstm = nn.LSTM(
            input_size=input_d, hidden_size=hidden_d, batch_first=True, num_layers=n_layers, **kwargs
        ).float()
        self.memory_h = NodeMemory(n_nodes, hidden_d, n_layers, device, memory_on_cpu)
        self.memory_c = NodeMemory(n_nodes, hidden_d, n_layers, device, memory_on_cpu)

    def reset_memory_state(self):
        """Reset the node memory for hidden states."""
        self.memory_h.reset_state()
        self.memory_c.reset_state()

    def forward(self, in_state: Tuple[torch.Tensor, List[int]]):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids = in_state
        if not self.mini_batch:
            node_embs = node_embs[ids]
        h = self.memory_h.get_memory(ids, self.device)
        c = self.memory_c.get_memory(ids, self.device)
        out_sequential, (h, c) = self.lstm(node_embs.view(-1, 1, self.input_d), (h, c))
        self.memory_h.update_memory(h.detach().clone(), ids)
        self.memory_c.update_memory(c.detach().clone(), ids)
        out = self.simple_decoder((out_sequential.view(-1, self.hidden_d), ids))
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
        device: Union[str, torch.device],
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
            device: str or torch.device, the device this model will run. Mainly for node memory.
            memory_on_cpu: bool, whether to store hidden state memory on RAM.
        """
        super().__init__(hidden_d, n_nodes, simple_decoder, device)
        self.input_d = input_d
        self.gru = nn.GRU(
            input_size=input_d, hidden_size=hidden_d, batch_first=True, num_layers=n_layers, **kwargs
        ).float()
        self.memory_h = NodeMemory(n_nodes, hidden_d, n_layers, device, memory_on_cpu)

    def reset_memory_state(self):
        """Reset the node memory for hidden states."""
        self.memory_h.reset_state()

    def forward(self, in_state: Tuple[torch.Tensor, List[int]]):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids = in_state
        if not self.mini_batch:
            node_embs = node_embs[ids]
        h = self.memory_h.get_memory(ids, self.device)
        out_sequential, h = self.gru(node_embs.view(-1, 1, self.input_d), h)
        self.memory_h.update_memory(h.detach().clone(), ids)
        out = self.simple_decoder((out_sequential.view(-1, self.hidden_d), ids))
        return out


class RNN(SequentialDecoder):
    """RNN sequential decoder."""

    def __init__(
        self,
        input_d: int,
        hidden_d: int,
        n_nodes: int,
        n_layers: int,
        simple_decoder: SimpleDecoder,
        device: Union[str, torch.device],
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
            device: str or torch.device, the device this model will run. Mainly for node memory.
            memory_on_cpu: bool, whether to store hidden state memory on RAM.
        """
        super().__init__(hidden_d, n_nodes, simple_decoder, device)
        self.input_d = input_d
        self.rnn = nn.RNN(
            input_size=input_d, hidden_size=hidden_d, batch_first=True, num_layers=n_layers, **kwargs
        ).float()
        self.memory_h = NodeMemory(n_nodes, hidden_d, n_layers, device, memory_on_cpu)

    def reset_memory_state(self):
        """Reset the node memory for hidden states."""
        self.memory_h.reset_state()

    def forward(self, in_state: Tuple[torch.Tensor, List[int]]):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids = in_state
        if not self.mini_batch:
            node_embs = node_embs[ids]
        h = self.memory_h.get_memory(ids, self.device)
        out_sequential, h = self.rnn(node_embs.view(-1, 1, self.input_d), h)
        self.memory_h.update_memory(h.detach().clone(), ids)
        out = self.simple_decoder((out_sequential.view(-1, self.hidden_d), ids))
        return out
