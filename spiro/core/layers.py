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
from spiro.core.backends import PYTORCH, TENSORFLOW
from spiro.core.errors import BackendNotSupportedError
from spiro.core.utils import _get_backend

this_backend = _get_backend()

if this_backend == TENSORFLOW:
    from spiro.core.tf.conv1D import Conv1d
    from spiro.core.tf.time2Vec import Time2Vec
    from spiro.core.tf.tsa import TSA

else:
    raise BackendNotSupportedError("Backend %s not supported." % (this_backend))
