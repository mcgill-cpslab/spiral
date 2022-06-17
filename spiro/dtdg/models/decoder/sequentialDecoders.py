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
from spiro.core.backends import PYTORCH, TENSORFLOW
from spiro.core.errors import BackendNotSupportedError
from spiro.core.utils import _get_backend

this_backend = _get_backend()

if this_backend == TENSORFLOW:
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import (
        FTSA,
        GRU,
        LSTM,
        LSTM_N,
        PTSA,
        RNN,
        Conv1D,
        Conv1D_N,
        NodeTrackingPTSA,
        SelfAttention,
    )
elif this_backend == PYTORCH:
    from spiro.dtdg.models.decoder.torch.sequentialDecoder.implicitTimeModels import GRU, LSTM, RNN
else:
    raise BackendNotSupportedError("Backend %s not supported." % (this_backend))
