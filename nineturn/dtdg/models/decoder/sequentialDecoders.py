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
"""Nine Turn specific errors and exceptions."""
# flake8: noqa
# Dynamic import, no need for lint
from nineturn.core.backends import PYTORCH, TENSORFLOW
from nineturn.core.errors import BackendNotSupportedError
from nineturn.core.utils import _get_backend

this_backend = _get_backend()

if this_backend == TENSORFLOW:
    from nineturn.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import (
        FTSA,
        GRU,
        LSTM,
        PTSA,
        NodeTrackingPTSA,
        RNN,
        Conv1D,
        SelfAttention,
    )
elif this_backend == PYTORCH:
    from nineturn.dtdg.models.decoder.torch.sequentialDecoder.implicitTimeModels import GRU, LSTM, RNN
else:
    raise BackendNotSupportedError("Backend %s not supported." % (this_backend))
