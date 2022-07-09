#!/usr/bin/env python
# flake8: noqa
"""Tests `spiro.dtdg.models.decoder.torch.sequentialDecoder.implicitTimeModels` package."""
import torch
from tests.core.common_functions import *


def test_node_memory_torch():
    """Test NodeMemory"""
    clear_background()
    from spiro.core.config import set_backend
    from spiro.core.backends import PYTORCH
    set_backend(PYTORCH)
    from spiro.dtdg.models.decoder.torch.sequentialDecoder.implicitTimeModels import NodeMemory

    n_nodes = 5
    hidden_d = 2
    n_layers = 3
    this_memory = NodeMemory(n_nodes,hidden_d,n_layers)
    nodes_to_change = [2,3]
    new_memory = torch.randn(n_layers, 2, hidden_d)
    this_memory.update_memory(new_memory, nodes_to_change)
    assert torch.equal(this_memory.get_memory(nodes_to_change), new_memory)
    old_memory = this_memory.memory.clone()
    this_memory.reset_state()
    assert not torch.equal(old_memory, this_memory.memory)


