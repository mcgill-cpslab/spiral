# ============================================================================
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
# mypy: ignore-errors
"""Pytorch based sequential decoder. Designed specially for dynamic graph learning."""

import copy
from typing import List, Tuple

import torch
import torch.nn as nn

from spiro.dtdg.models.decoder.torch.simpleDecoder import SimpleDecoder


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
        self.memory = torch.randn(self.n_layers, self.n_nodes, self.hidden_d)

    def update_memory(self, new_memory: torch.Tensor, inx: List[int]):
        """Update memory."""
        self.memory[:, inx, :] = new_memory

    def get_memory(self, inx: List[int]) -> torch.Tensor:
        """Retrieve node memory by index."""
        return self.memory[:, inx, :]

    @property
    def device(self):
        """Which device it is currently at."""
        return self.memory.device

    def to(self, device, **kwargs):  # pylint: disable_invalide_name
        """Move the snapshot to the targeted device (cpu/gpu).

        If the graph is already on the specified device, the function directly returns it.
        Otherwise, it returns a cloned graph on the specified device.

        Args:
            device : Framework-specific device context object
                The context to move data to (e.g., ``torch.device``).
            kwargs : Key-word arguments.
                Key-word arguments fed to the framework copy function.
        """
        if device is None or self.device == device:
            return self

        ret = copy.copy(self)
        ret.memory = self.memory.to(device, **kwargs)
        return ret


class SequentialDecoder(nn.Module):
    """Prototype of sequential decoders."""

    def __init__(self, hidden_d: int, n_nodes: int, n_layers: int, simple_decoder: SimpleDecoder):
        """Create a sequential decoder.

        Args:
            hidden_d: int, the hidden state's dimension.
            n_nodes: int, number of nodes to remember.
            n_layers: int, number of targeting rnn layers.
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_d = hidden_d
        self.mini_batch = False
        self.simple_decoder = simple_decoder
        self.memory_h = NodeMemory(n_nodes, hidden_d, n_layers)
        self.training_mode = True
        self.base_model = None

    def set_mini_batch(self, mini_batch: bool = True):
        """Set to batch training mode."""
        self.mini_batch = mini_batch

    @property
    def device(self):
        """Which device it is currently at."""
        return next(self.base_model.parameters()).device

    def to(self, device, **kwargs):  # pylint: disable_invalide_name
        """Move the snapshot to the targeted device (cpu/gpu).

        If the graph is already on the specified device, the function directly returns it.
        Otherwise, it returns a cloned graph on the specified device.

        Args:
            device : Framework-specific device context object
                The context to move data to (e.g., ``torch.device``).
            kwargs : Key-word arguments.
                Key-word arguments fed to the framework copy function.
        """
        if device is None or self.device == device:
            return self

        ret = copy.copy(self)
        ret.base_model = self.base_model.to(device, **kwargs)
        ret.simple_decoder = self.simple_decoder.to(device, **kwargs)
        ret.memory_h = self.memory_h.to(device, **kwargs)
        return ret

    def training(self):
        """Start training."""
        self.training_mode = True

    def eval_mode(self):
        """Stop training."""
        self.training_mode = False

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
        h = self.memory_h.get_memory(ids)
        out_sequential, h = self.base_model(node_embs.view(-1, 1, self.input_d), h)
        if self.training_mode:
            self.memory_h.update_memory(h.detach().clone(), ids)
        out = self.simple_decoder((out_sequential.view(-1, self.hidden_d), ids))
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
        self.base_model = nn.LSTM(
            input_size=input_d, hidden_size=hidden_d, batch_first=True, num_layers=n_layers, **kwargs
        ).float()
        self.memory_c = NodeMemory(n_nodes, hidden_d, n_layers)

    def reset_memory_state(self):
        """Reset the node memory for hidden states."""
        self.memory_h.reset_state()
        self.memory_c.reset_state()

    def to(self, device, **kwargs):  # pylint: disable_invalide_name
        """Move the snapshot to the targeted device (cpu/gpu).

        If the graph is already on the specified device, the function directly returns it.
        Otherwise, it returns a cloned graph on the specified device.

        Args:
            device : Framework-specific device context object
                The context to move data to (e.g., ``torch.device``).
            kwargs : Key-word arguments.
                Key-word arguments fed to the framework copy function.
        """
        if device is None or self.device == device:
            return self

        ret = super(self).to(device)
        ret.memory_c = self.memory_c.to(device, **kwargs)
        return ret

    def forward(self, in_state: Tuple[torch.Tensor, List[int]]):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids = in_state
        if not self.mini_batch:
            node_embs = node_embs[ids]
        h = self.memory_h.get_memory(ids)
        c = self.memory_c.get_memory(ids)
        out_sequential, (h, c) = self.base_model(node_embs.view(-1, 1, self.input_d), (h, c))
        if self.training_mode:
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
        self.base_model = nn.GRU(
            input_size=input_d, hidden_size=hidden_d, batch_first=True, num_layers=n_layers, **kwargs
        )


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
        self.base_model = nn.RNN(
            input_size=input_d, hidden_size=hidden_d, batch_first=True, num_layers=n_layers, **kwargs
        )
