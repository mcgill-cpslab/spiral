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
"""Model assembler for pytorch."""
from torch.nn import Module


class Assembler(Module):
    """Assembler combines encoder and decoder to create a dynamic graph learner."""

    def __init__(self, encoder, decoder):
        """Initialize the assembler.

        Args:
            encoder: graph encoder
            decoder: graph decoder
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_state):
        """Forward function."""
        h = self.encoder(input_state)
        h = self.decoder(h)
        return h
