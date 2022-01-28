#!/usr/bin/env python
# flake8: noqa
"""Tests `nineturn.dtdg.models.decoder.torch.sequentialDecoder.rnnFamily` package."""
import torch
from tests.core.common_functions import *


def test_ogb_dataset_torch():
    clear_background()
    """Test that citation graph could support different backend."""
    from nineturn.core.config import set_backend
    from nineturn.core.backends import PYTORCH
    set_backend(PYTORCH)
    from nineturn.dtdg.models.decoder.torch.sequentialDecoder.rnnFamily import NodeMemory

    n_nodes = 5
    hidden_d = 2
    n_layers = 3
    this_memory = NodeMemory(n_nodes,hidden_d,n_layers,"cpu")
    old_memory = this_memory.memory.clone()
    this_memory.reset_state()
    assert not torch.equal(old_memory, this_memory.memory)
    nodes_to_change = [2,3]
    new_memory = torch.randn(n_layers, 2, hidden_d)
    this_memory.update_memory(new_memory, nodes_to_change)
    assert torch.equal(this_memory.get_memory(nodes_to_change, "cpu"), new_memory)


