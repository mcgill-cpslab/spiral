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

    def get_weights(self):
        return [self.base_model.get_weights(), self.simple_decoder.get_weights()]

    def set_weights(self, weights):
        self.base_model.set_weights(weights[0])
        self.simple_decoder.set_weights(weights[1])
