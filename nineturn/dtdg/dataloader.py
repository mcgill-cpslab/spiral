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
"""Datasets and dataloader for DTDG experiments"""

import sys
from typing import Dict, List

import numpy as np
from numpy import ndarray
from ogb.linkproppred import LinkPropPredDataset

from nineturn.core.config import get_backend, get_logger
from nineturn.dtdg.types import CitationGraph, DiscreteGrapha

logger = get_logger()

EDGES = "edge_index"
NODES = "node_feat"


def preprocess_citation_graph(graph_data: Dict[str, ndarray], node_time: str) -> CitationGraph:
    """Data preprocessing for citation graph.

    Citation graph is an V-E invariant DTDG which has number of citations for each node changes every year.

    Args:
        graph_data: a dictionary with key edge_index corresponding to edge data asnumpy.ndarray of shape (2,|E|).
          Key node_index corresponding to node feature table. Random key corresponding to the timestamps the node is
          added with shape (1, |V|)
        node_time: a str for the random key for the node time.

    Returns:
        A CitationGraph instance
    """
    timestamps = np.unique(graph_data[node_time])
    node_time_index = np.searchsorted(timestamps, graph_data[node_time][:, 0])
    node_time_index = np.reshape(node_time_index, (graph_data[NODES].shape[0], 1))
    node_id = np.array(range(graph_data[NODES].shape[0]))
    node_id = np.reshape(node_id, (graph_data[NODES].shape[0], 1))
    nodes = np.hstack((node_id, node_time_index, graph_data[NODES]))
    nodes = nodes[nodes[:, 1].argsort()]  # [id,t,feat..]]
    nodes_id_dict = {A: B for A, B in zip(nodes[:, 0], np.array(range(nodes.shape[0])))}
    sources = np.array([nodes_id_dict[s] for s in graph_data[EDGES][0]])
    dest = np.array([nodes_id_dict[s] for s in graph_data[EDGES][1]])
    edge_time = np.array([nodes[s[0]][1] for s in sources])
    edges = np.vstack((edge_time, sources, dest)).transpose()  # [t,s,d]
    return CitationGraph(edges, nodes, timestamps)


OGB_DATASETS = {'ogbl-citation2': (preprocess_citation_graph, 'node_year')}
# A dict with dataset name as key, the tuple of preporcess method and time feature name as value


def supported_ogb_datasets() -> List[str]:
    """Return the list of supported ogb datasets.

    Returns:
        A list of obg dataset names
    """
    return OGB_DATASETS.keys()


def ogb_dataset(name: str) -> DiscreteGraph:
    """Download, preprocess the input OGB dataset.

    This function


    """
    if name not in OGB_DATASETS:
        logger.error("Dataset %s is not supported." % (name))
        raise SystemExit('Fatal error happens!')

    dataset = LinkPropPredDataset(name=name)[0]
    preprocess, time_feat = OGB_DATASETS[name]
    this_graph = preprocess(dataset, time_feat)
    return this_graph
