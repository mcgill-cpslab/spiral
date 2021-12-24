# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

from dgl import DGLGraph


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

    def __init__(self, observation: DGLGraph, t: int):
        """A snapshot of a DTDG composed by an instance of DGLGraph as observation and an integer as timestamp."""
        self.observation = observation
        self.t = t


class DiscreteGraph:
    """Discrete Time Dynamic Graph is a collection of static_graph as its snapshots and their attached timestamps."""

    def __init__(self, snapshots: list[Snapshot]):
        """A DTDG is a list of snapshots sorted by their timestamps asc."""
        self.snapshots = sorted(snapshots, key=lambda x: x.t)
