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
"""Pytorch based simple decoders. Designed specially for dynamic graph learning."""

from abc import abstractmethod
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from spiro.core.errors import DimensionError
from spiro.core.logger import get_logger
from spiro.core.types import MLBaseModel, Tensor, nt_layers_list

logger = get_logger()


class SimpleDecoder(MLBaseModel):
    """Prototype of simple decoder."""

    def __init__(self):
        """Init function."""
        super().__init__()
        self.nn_layers = nt_layers_list()

    @abstractmethod
    def call(self, in_sample: Tuple[Tensor, List]) -> Tensor:
        """All SimpleDecoder subclass should have a forward function.

        Args:
            in_sample: tuple, first entry is a nodes embedding and second entry is the list[int] of targeted node ids.

        Return:
            prediction: Tensor, the prediction.
        """
        pass


class MLP(SimpleDecoder):
    """Multi layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        embed_dims: List[int],
        dropout: float = 0.5,
        output_dim: int = 1,
        activation: str = "linear",
    ):
        """Init function.

        Args:
            input_dim: int, input dimension.
            embed_dims: list of int, indicating the dimension or each layer.
            dropout: float, dropout rate.
            output_dim: int, number of class in output.
            activation: str, name of the activation function. must be one supported by the tensorflow dense layer
        """
        super().__init__()
        for embed_dim in embed_dims:
            self.nn_layers.append(layers.Dense(embed_dim))
            self.nn_layers.append(layers.BatchNormalization())
            self.nn_layers.append(layers.ReLU())
            self.nn_layers.append(layers.Dropout(dropout))
        self.nn_layers.append(layers.Dense(output_dim, activation=activation))

    def call(self, in_state):
        """Implementation of forward.

        Args:
            in_state: tuple, first entry is either a nodes embedding or the hidden representation from a sequential
                       decoder, second entry is the list[int] of targeted node ids or a list[[int, int]] for edge
                       prediction.

        Return:
            prediction: Tensor
        """
        emb, ids_in = in_state
        ids_rank = tf.rank(ids_in).numpy()
        if ids_rank == 1:
            mlp_h = emb
        elif ids_rank == 2:
            n_edges = ids_in.shape[0]
            ids_id = tf.reshape(ids_in, [-1])
            mlp_h = tf.reshape(tf.gather(emb, ids_id), [n_edges, -1])
        else:
            message = f"""The index to predict in the input must be of rank 1 for node prediction or rank 2 for edge
            prediction. But get an input index of rank {ids_rank}"""
            logger.error(message)
            raise DimensionError(message)
        for layer in self.nn_layers:
            mlp_h = layer(mlp_h)

        return mlp_h
