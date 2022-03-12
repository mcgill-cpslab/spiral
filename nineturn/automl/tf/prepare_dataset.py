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
"""Prepare dataset for different learning tasks."""

from typing import Dict, List, Tuple
import numpy as np
from nineturn.core.dataio import neg_sampling
from nineturn.dtdg.types import VEInvariantDTDG, Snapshot
from nineturn.core.errors import DimensionError

TARGET = 'target'
LABEL = 'label'



def get_new_nodes(dgraph: VEInvariantDTDG, t:int) -> np.ndarray:
    new_nodes =np.arange(dgraph._node_time_anchors[t-1], dgraph._node_time_anchors[t])
    return new_nodes


def edge_sample(snapshot: Snapshot, target_node:int, num_positive_edges:int, num_negative_edges:int=None) -> Tuple[List,List]:
    if num_negative_edges is None:
        num_negative_edges = num_positive_edges

    this_graph= snapshot.observation
    neighbours = this_graph.successors(target_node).numpy()
    re = ([],[])
    if len(neighbours) > num_positive_edges:
        pos_inds = np.append(neighbours, target_node)
        neg_sample = neg_sampling(pos_inds, this_graph.num_nodes(), num_negative_edges)
        pos_id = np.random.randint(0,len(pos_inds)-1,num_positive_edges)
        pos_sample = neighbours[pos_id]
        re = ([[target_node, i] for i in pos_sample],[[target_node, i] for i in neg_sample])
    return re

def prepare_edge_task(dgraph: VEInvariantDTDG, num_postive:int, num_negative:int=None, start_t:int=1):
    times = len(dgraph)
    dgraph.time_data[TARGET] = {}
    dgraph.time_data[LABEL] = {}
    dgraph.time_data['positive_edges'] = {}
    for t in range(start_t, times):
        snapshot,all_nodes = dgraph.dispatcher(t)
        pos = []
        neg = []
        labels = []
        nodes = list(range(dgraph._node_time_anchors[t-1], dgraph._node_time_anchors[t]))
        for n in nodes:
            edges = edge_sample(snapshot, n, num_postive,num_negative)
            pos += edges[0]
            neg += edges[1]
        
        edge_to_re = [dgraph.edge_id[f"{e[0]}_{e[1]}"] for e in pos]
        dgraph.time_data['positive_edges'][t] = edge_to_re
        edge_target = pos + neg
        dgraph.time_data[TARGET][t] = np.array(edge_target)
        if len(pos) < 1 or len(neg) < 1:
            raise DimensionError(f"positive and negative sample has {len(pos)} and {len(neg)} edges") 
        labels = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
        dgraph.time_data[LABEL][t] = labels
