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
"""common types for the dtdg package.

This file define the types required for dtdg package
"""
import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import dgl
import numpy as np
from dgl import DGLGraph
from dgl import backend as F
from numpy import ndarray

from spiro.core import commonF
from spiro.core.errors import DimensionError, ValueNotSortedError
from spiro.core.logger import get_logger
from spiro.core.types import Tensor
from spiro.core.utils import get_anchor_position, is_sorted

TIME_D = 0  # position of time in nodes and edges table
SOURCE = 1  # position of source in edges table
DESTINATION = 2  # position of destination in edges_table
FEAT = 'h'  # key for edge and node feature in DGLGraph.edata and ndata.
ID_TYPE = 'int64'
FEATURE_TYPE = "float32"
logger = get_logger()


class Snapshot:
    """A snapshot of a dynamic graph.

    The snapshot is usually a tuple (V,X,E,t) where X is the node feature table,
    V is the adjacency matrix, E is the edge feature table with the first entry the source id and the second entry the
    destination id. And edge e in E could have more than two entry. The extra ones would be edge features.
    t is the timestamp when the snapshot was taken. For DTDG, this is usually an integer representing the positional
    ordering.
    In this implementation, the graph state (V,X,E) is implemented by a DGLGraph.
    When designing this class, our primary goal is to support the loading of dynamic graph data
    in 'https://snap.stanford.edu/data/',
    """

    def __init__(self, observation: DGLGraph, t: int, node_ids: ndarray = None):
        """A snapshot of a DTDG composed by an instance of DGLGraph as observation and an integer as timestamp."""
        self.observation = observation
        self.t = commonF.to_tensor(np.array([t]))
        self.node_ids = None
        if node_ids:
            self.node_ids = commonF.to_tensor(node_ids)

    def num_node_features(self) -> int:
        """Return the number of node features."""
        return self.observation.ndata[FEAT].shape[1]

    def num_edge_features(self) -> int:
        """Return the number of edge features."""
        return self.observation.edata[FEAT].shape[1]

    def node_feature(self):
        """Return the node features tensor."""
        return self.observation.ndata[FEAT]

    def edge_feature(self):
        """Return the edge feature tensor."""
        return self.observation.edata[FEAT]

    def num_nodes(self) -> int:
        """Return the number of nodes in the snapshot."""
        return self.observation.ndata[FEAT].shape[0]

    @property
    def device(self):
        """Which device it is currently at."""
        return self.observation.device

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
        ret.observation = self.observation.to(device, **kwargs)
        ret.t = F.copy_to(self.t, device, **kwargs)
        return ret


class DiscreteGraph(ABC):
    """This is an abstract class for Discrete Time Dynamic Graph collection.

    In implementation, we found that there could be four different kinds of DTDG, V-E invariant, V invariant,
    E invariant, and V_E variant DTDG. Even though they all can be expressed as collection of timestamped snapshots.
    However, it is memory inefficient to do so. Therefore, we need to have different methods to store different
    type of DTDG, and consequently, different type of DTDG requires different data dispatcher.
    As a subclass of DiscreteGraph, it needs to implement its own data dispatcher to generate snapshots in runtime.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total of observations in the graph."""
        pass

    @abstractmethod
    def dispatcher(self, t: int) -> Tuple[Snapshot, List]:
        """Return the snapshot observed at the input time index."""
        pass


class VEInvariantDTDG(DiscreteGraph):
    """V-E invariant DTDG.

    V-E invariant DTDG is a DTDG that nodes and edges won't change in terms of features or their existence after they
    are created in the graph.
    """

    def __init__(self, edges: ndarray, nodes: ndarray, timestamps: ndarray, node_names: ndarray = None):
        """V-E invariant DTDG is stored as an edge table,a node table an the timestamp index.

        Args:
            edges: a numpy.ndarray of edges with shape (|E|,3+num_features). Each row is an edge with the first entry to
              be the timestamp index,the second and the third entry to be the source and destination index corresponding
              to the row index in the nodes table.
            nodes: a numpy.ndarray of node features with shape (|V|, 1+num_features). Each row is a node and the first
              entry is the timestamp index.
            timestamps: a numpy.ndarray of timestamps with shape (1, num_observations). This array should be sorted
              asc by the timestamps.

        Raises:
            DimensionError if the input edge, nodes or timestamps has a different dimension than expected.
            ValueNotSortedError if the input edges, nodes or timestamps is not sorted based on the time dimension.
        """
        self.edge_dimension = 3
        self.node_dimension = 1
        self.node_names = node_names
        error_message = """The second dimension of {entity} should be greater than or equal to {value}."""
        not_sorted_error = """The input {entity} should be sorted based on the its time index."""
        if edges.shape[1] < self.edge_dimension:
            raise DimensionError(error_message.format(entity="edges", value=self.edge_dimension))

        if nodes.shape[1] < self.node_dimension:
            raise DimensionError(error_message.format(entity="nodes", value=self.node_dimension))

        if not is_sorted(timestamps):
            raise ValueNotSortedError(not_sorted_error.format(entity="timestamps"))

        if not is_sorted(edges[:, TIME_D]):
            raise ValueNotSortedError(not_sorted_error.format(entity="edges"))

        if not is_sorted(nodes[:, TIME_D]):
            raise ValueNotSortedError(not_sorted_error.format(entity="nodes"))

        self.nodes = nodes
        self.edges = edges
        self.timestamps = timestamps
        self._node_time_anchors = get_anchor_position(nodes[:, TIME_D], range(len(self.timestamps)))
        self._edge_time_anchors = get_anchor_position(edges[:, TIME_D], range(len(self.timestamps)))
        self.edge_id = {f"{int(self.edges[i][1])}_{int(self.edges[i][2])}": i for i in range(len(self.edges))}
        self.time_data: Dict = {}

    def dispatcher(self, t: int) -> Tuple[Snapshot, Tensor]:
        """Return a snapshot for the input time index. Time index start from 0, end at num_snapshot - 1."""
        this_edges = self.edges[: self._edge_time_anchors[t], :]
        this_nodes = self.nodes[: self._node_time_anchors[t], :]
        num_nodes = this_nodes.shape[0]
        node_ids = np.arange(num_nodes)
        src = commonF.to_tensor(this_edges[:, SOURCE].astype(ID_TYPE))
        dst = commonF.to_tensor(this_edges[:, DESTINATION].astype(ID_TYPE))
        observation = dgl.graph((src, dst), num_nodes=num_nodes)
        if this_edges.shape[1] > self.edge_dimension:
            observation.edata[FEAT] = commonF.to_tensor(this_edges[:, self.edge_dimension :].astype(FEATURE_TYPE))

        if this_nodes.shape[1] > self.node_dimension:
            observation.ndata[FEAT] = commonF.to_tensor(this_nodes[:, self.node_dimension :].astype(FEATURE_TYPE))

        return (Snapshot(observation, t), commonF.to_tensor(node_ids))

    def __len__(self) -> int:
        """Return the number of snapshots in this DTDG."""
        return len(self.timestamps)


class CitationGraph(VEInvariantDTDG):
    """Citation graph is a V-E invariant DTDG.

    A citation graph is roughly a V-E invariant DTDG, it is different from other kinds of dynamic graph
    in that each node's citation increase over time. Besides citations, other features remain the same.
    """

    def __init__(self, edges: ndarray, nodes: ndarray, timestamps: ndarray):
        """Citation graph is a subclass of VEInvariantDTDG..

        Args:
            edges: a numpy.ndarray of edges with shape (|E|,3+num_features). Each row is an edge with the first entry to
              be the timestamp index,the second and the third entry to be the source and destination index corresponding
              to the row index in the nodes table.
            nodes: a numpy.ndarray of node features with shape (|V|, 1+num_features). Each row is a node and the first
              entry is the timestamp index.
            timestamps: a numpy.ndarray of timestamps with shape (1, num_observations). This array should be sorted
              asc by the timestamps.

        Raises:
            DimensionError if the input edge, nodes or timestamps has a different dimension than expected.
            ValueNotSortedError if the input edges, nodes or timestamps is not sorted based on the time dimension.
        """
        super().__init__(edges, nodes, timestamps)

    def dispatcher(self, t: int, add_self_loop: bool = False) -> Tuple[Snapshot, Tensor]:
        """Return a snapshot for the input time index.

        For citation graph, the node feature has previous year's citation as the last node feature.
        """
        current_snapshot, node_ids = super().dispatcher(t)
        observation = current_snapshot.observation
        this_nodes = observation.ndata[FEAT].numpy()
        citation = np.zeros(shape=(this_nodes.shape[0], 1))
        if t > 0:
            previous_snapshot, _ = super().dispatcher(t - 1)
            previous_citation = previous_snapshot.observation.in_degrees().numpy()
            citation[: previous_citation.shape[0], 0] = previous_citation

        this_nodes = np.hstack((this_nodes, citation))
        observation.ndata[FEAT] = commonF.to_tensor(this_nodes[:, self.node_dimension :].astype(FEATURE_TYPE))
        if add_self_loop:
            observation = observation.add_self_loop()
        return (Snapshot(observation, t), node_ids)
