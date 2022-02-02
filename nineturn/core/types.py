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
"""Dynamic import common functions based on backend."""
# flake8: noqa
# Dynamic import, no need for lint
from nineturn.core.backends import PYTORCH, TENSORFLOW
from nineturn.core.errors import BackendNotSupportedError
from nineturn.core.utils import _get_backend

this_backend = _get_backend()

def nt_layers_list():
    if this_backend == TENSORFLOW:
        return []
    else:
        return 



if this_backend == TENSORFLOW:
    from tensorflow import Tensor
    from tensorflow.keras.layers import Layer as MLBaseModel, Dropout as Dropout
    from tensorflow.keras import Sequential
    from nineturn.core.tf_functions import nt_layers_list
    from dgl.nn.tensorflow.conv import GATConv, GraphConv, SAGEConv, SGConv

elif this_backend == PYTORCH:
    from torch.nn import Module as MLBaseModel, Sequential as Sequential, Dropout as Dropout
    from torch import Tensor as Tensor
    from nineturn.core.torch_functions import nt_layers_list
    from dgl.nn.pytorch.conv import GATConv, GraphConv, SAGEConv, SGConv

else:
    raise BackendNotSupportedError("Backend %s not supported." % (this_backend))
