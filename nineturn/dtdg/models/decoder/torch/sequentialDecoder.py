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
from typing import Tuple, Union,List
import torch
import torch.nn as nn
from nineturn.dtdg.models.decoder.torch.simpleDecoder import SimpleDecoder

class NodeMemory:
    """NodeMemory to remember states for each node."""

    def __init__(self, n_nodes:int, hidden_d:int,device:Union[str,torch.device], memory_on_cpu:bool=False):
        """Create a node memory based on the number of nodes and the state dimension.

        Args:
            n_nodes: int, number of nodes to remember.
            hidden_d: int, the hidden state's dimension.
        """
        self.n_nodes = n_nodes
        self.hidden_d = hidden_d
        self.device = device 
        if memory_on_cpu:
            self.device = "cpu"
        self.reset_state()
    def reset_state(self):
        """Reset the memory to a random tensor."""
        self.memory_h = torch.randn(self.n_nodes, 1, self.hidden_d).to(self.device)
        self.memory_c = torch.randn(self.n_nodes, 1, self.hidden_d).to(self.device)

    def update_memory(self, new_memory_h, new_memory_c, inx):
        """Update memory."""
        self.memory_h[inx] = new_memory_h[0].view(-1, 1, self.hidden_d).to(self.device)
        self.memory_c[inx] = new_memory_c[0].view(-1, 1, self.hidden_d).to(self.device)

    def get_memory(self, inx, device):
        """Retrieve node memory by index."""
        return (self.memory_h[inx].view(1, -1, self.hidden_d).to(device), self.memory_c[inx].view(1, -1,
            self.hidden_d).to(device))


class SequentialDecoder(nn.Module):
    """Prototype of sequential decoders."""
    
    def __init__(self, hidden_d:int, n_nodes:int, simple_decoder:SimpleDecoder,device:Union[str,torch.device], memory_on_cpu:bool=False):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_d = hidden_d
        self.memory = NodeMemory(n_nodes, hidden_d,device,memory_on_cpu)
        self.simple_decoder = simple_decoder

    @abstractmethod
    def forward(self, in_sample: Tuple[torch.Tensor, List]) -> torch.Tensor:
        """All SequentialDecoder subclass should have a forward function.

        Args:
            in_sample: tuple, first entry is a node-wise embedding from an encoder, second entry is the list[int] of targeted node ids.


        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        pass


class LSTM(SequentialDecoder):
    """LSTM sequential decoder."""

    def __init__(self, input_d:int, hidden_d:int, n_nodes:int, simple_decoder:SimpleDecoder,
            device:Union[str,torch.device], memory_on_cpu:bool=False):
        """Create a LSTM sequential decoder.

        Args:
            input_d: int, input dimension.
            hidden_d: int, number of hidden cells.
            n_nodes: int, number of nodes.
            simple_decoder: an instance of SimpleDecoder.
            device: str or torch.device, the device this model will run. Mainly for node memory.
        """
        super().__init__(hidden_d,n_nodes,simple_decoder,device,memory_on_cpu)
        self.input_d = input_d
        self.lstm = nn.LSTM(input_size=input_d, hidden_size=hidden_d, batch_first=True, num_layers=1).float()
        self.mini_batch = False
        self.device = device

    def set_mini_batch(self, mini_batch: bool = True):
        """Set to batch training mode."""
        self.mini_batch = mini_batch

    def forward(self, in_state:Tuple[torch.Tensor, List[int]]):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids = in_state
        if not self.mini_batch:
            node_embs = node_embs[ids]
        h, c = self.memory.get_memory(ids, self.device)
        out_sequential, (h, c) = self.lstm(node_embs.view(-1, 1, self.input_d), (h, c))
        out = self.simple_decoder((out_sequential.view(-1,self.hidden_d), ids))
        self.memory.update_memory(h.detach().clone(), c.detach().clone(), ids)
        return out
