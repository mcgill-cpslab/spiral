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

import numpy as np
from tensorflow.keras import layers

from nineturn.dtdg.models.decoder.tf.simpleDecoder import SimpleDecoder


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
        self.training_mode = True

    def eval_mode(self):
        self.training_mode = False

    def set_mini_batch(self, mini_batch: bool = True):
        """Set to batch training mode."""
        self.mini_batch = mini_batch


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
        self.memory = np.zeros((self.n_nodes, self.window_size, self.input_d))

    def update_window(self, new_window, inx):
        """Update memory with input memory [N,D].
        Args:
            new_window: numpy array,
        """
        self.memory[inx, :-1, :] = self.memory[inx, 1:, :]
        self.memory[inx, -1, :] = new_window

    def get_memory(self, inx):
        """Retrieve node memory by index.Return shape [N,W,D]."""
        return self.memory[inx]


class SlidingWindowFamily(BaseModel):
    """Prototype of sliding window based sequential decoders."""

    def __init__(self, input_d: int, n_nodes: int, window_size: int, simple_decoder: SimpleDecoder):
        """Create a sequential decoder.

        Args:
            input_d: int, the hidden state's dimension.
            window_size: int, the length of the sliding window
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        super().__init__(input_d, simple_decoder)
        self.window_size = window_size
        self.training_mode = True
        self.n_nodes = n_nodes
        self.memory = SlidingWindow(self.n_nodes, self.hidden_d, self.window_size)
