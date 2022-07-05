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
"""Assemble a dynamic graph learning model."""
import os
import numpy
from tensorflow import keras
from spiro.core.logger import get_logger

logger = get_logger()


class Assembler(keras.Model):
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

    def call(self, input_state, training=False):
        """Forward function."""
        h = self.encoder(input_state, training)
        h = self.decoder(h, training)
        return h

    def save_model(self, path: str):
        """Save the model into the input path."""
        if not os.path.exists(path):
            os.mkdir(path)
        if self.encoder.has_weights():
            encoder_weights = self.encoder.get_weights()
            numpy.save(f"{path}/encoder", encoder_weights)
        decoder_weights = self.decoder.get_weights()
        numpy.save(f"{path}/decoder", decoder_weights)

    def load_model(self, path):
        """Load encoder-decoder model from path."""
        if self.encoder.has_weights():
            encoder_weights = numpy.load(f"{path}/encoder.npy", allow_pickle=True)
            self.encoder.set_weights(encoder_weights)
        decoder_weights = numpy.load(f"{path}/decoder.npy", allow_pickle=True)
        self.decoder.set_weights(decoder_weights)
