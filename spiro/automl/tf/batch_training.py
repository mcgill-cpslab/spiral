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
"""Batch training related."""
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from spiro.automl.tf.prepare_dataset import LABEL, TARGET
from spiro.automl.tf.sampler import multi_hop_sampler
from spiro.core.errors import ValueError
from spiro.core.logger import get_logger
from spiro.dtdg.types import DiscreteGraph, Snapshot

logger = get_logger()


class NodeBatchGenerator(Sequence):
    """Batch Generator for batch training."""

    def __init__(
        self,
        dgraph: DiscreteGraph,
        batch_size: int,
        start_t: int,
        end_t: int,
        shuffle: bool = True,
        hops: int = 3,
        fanout: int = 10,
        edge_dir: str = 'in',
    ):
        """Init method.

        Args:
            dgraph: DiscreteGraph, a DTDG to be learned
            batch_size: int, the batch size
            start_t: int, the starting timestamp to perform sampling
            end_t: int, the ending timestamp to perform sampling
            shuffle: bool, shuffle the data in each iteration
            hops: int = 3, number of hops to sample from
            fanout: int = 10
            edge_dir: str = 'in', the direction of edges to sample from
        """
        self.dgraph = dgraph
        self.check_data()
        self.batch_size = batch_size
        self.start_t = start_t
        self.end_t = end_t
        self.target_nodes = np.array([])
        self.labels = np.array([])
        self.timestamps = np.array([])
        self.shuffle = shuffle
        self.on_epoch_end()
        self.hops = hops
        self.fanout = fanout
        self.edge_dir = edge_dir

    def check_data(self):
        """Check if the input DTDG is prepared for batch generation."""
        if not self.dgraph.time_data[TARGET] or not self.dgraph.time_data[LABEL]:
            error_message = """BatchGenerator should take a prepared dataset to generate batches. Please prepare the
            dataset as shown in the example."""
            logger.error(error_message)
            raise ValueError(error_message)

    def __len__(self) -> int:
        """Return the number of batches in this generator."""
        total_batches = 0
        for t in range(self.start_t, self.end_t):
            total_batches += math.ceil(len(self.dgraph.time_data[TARGET][t]) / self.batch_size)
        return total_batches

    def __next__(self):
        """Get the next batch."""
        idx = self.idx
        if idx < self.__len__():
            targets = tf.convert_to_tensor(self.target_nodes[idx], dtype=tf.int64)
            labels = tf.convert_to_tensor(self.labels[idx])
            t = self.timestamps[idx]
            this_graph = self.dgraph.dispatcher(t)[0].observation
            reduced_graph = multi_hop_sampler(this_graph, targets, self.hops, self.fanout, self.edge_dir)
            new_snapshot = Snapshot(reduced_graph, t)
            self.idx = idx + 1
            return (new_snapshot, targets, labels)
        else:
            raise StopIteration

    def __iter__(self):
        """Required for an iterator."""
        return self

    def on_epoch_end(self):
        """Generate target_nodes and timestamps for idx access."""
        self.idx = 0
        batch_size = self.batch_size
        if self.shuffle or self.target_nodes.size == 0:
            this_batch_node = []
            this_batch_label = []
            this_batch_time = []
            for t in range(self.start_t, self.end_t):
                current_t = self.dgraph.time_data[TARGET][t].numpy()
                current_label = self.dgraph.time_data[LABEL][t].numpy()
                new_sequence = np.dstack((current_t, current_label))[0]
                np.random.shuffle(new_sequence)
                total_nodes_t = current_t.shape[0]
                current_id = 0
                start_idx = current_id * batch_size
                end_idx = (current_id + 1) * batch_size
                while end_idx < total_nodes_t:
                    current_sequence = new_sequence[start_idx:end_idx]
                    this_batch_node.append(np.array([i[0] for i in current_sequence]))
                    this_batch_label.append(np.array([i[1] for i in current_sequence]))
                    this_batch_time.append(t)
                    current_id += 1
                    end_idx = (current_id + 1) * batch_size
                    start_idx = current_id * batch_size
                this_batch_node.append(current_t[start_idx:])
                this_batch_label.append(current_label[start_idx:])
                this_batch_time.append(t)
            self.target_nodes = np.array(this_batch_node)
            self.labels = np.array(this_batch_label)
            self.timestamps = np.array(this_batch_time)
