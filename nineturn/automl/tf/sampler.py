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
"""Graph sampling."""

import dgl
from dgl import DGLGraph
from dgl.sampling.neighbor import sample_neighbors
import tensorflow as tf
from tensorflow import Tensor
from nineturn.core.errors import DimensionError


def multi_hop_sampler(g: DGLGraph, seed:Tensor,  hops: int= 3, fanout:int = 10,edge_dir: str = 'in', **kwargs) -> DGLGraph:
    if len(seed.shape) == 1:
        target_nodes = seed
    elif len(seed.shape) == 2:
        pass
    else:
        error_message = f"""
                Expecting an input target_nodes is a rank 1 or rank 2 tensor, but get a tensor of rank {target_nodes.rank}.
            """
        logger.error(error_message)
        raise DimensionError(error_message)
    for i in range(hops-1):
        edges = sample_neighbors(g, target_nodes, edge_dir=edge_dir,fanout=fanout, **kwargs).edges()
        target_nodes,_  = tf.unique(tf.concat((edges[0],edges[1],target_nodes),0))
    return sample_neighbors(g, target_nodes, edge_dir=edge_dir,fanout=fanout, **kwargs)



