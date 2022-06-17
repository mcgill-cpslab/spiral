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
"""helper function to set runtime  backend.

This module contains the function to define the runtime context which include backend.

Example:
    >>> from spiro.core.config import set_backend
    >>> set_backend('tensorflow')
    >>> import dgl
"""
import os

from spiro.core.backends import PYTORCH, TENSORFLOW, supported_backends
from spiro.core.logger import get_logger

if '_BACKEND' not in globals():
    _BACKEND = None
_NINETURN_BACKEND = "NINETURN_BACKEND"
_DGL_BACKEND = "DGLBACKEND"
_TENSORFLOW = TENSORFLOW
_PYTORCH = PYTORCH
_BACKEND_NOT_FOUND = f"""
    Nine Turn backend is not selected or invalid. Assuming {_PYTORCH} for now.
    Please set up environment variable '{_NINETURN_BACKEND}' to one of '{_TENSORFLOW}'
    and '{_PYTORCH}' to select your backend.
    """
_BACKEND_NOT_SET = f"""Nine Turn backend is not set in either pipeline code, configuration file
    or environment variable. Assuming {_PYTORCH} for now.
    """

logger = get_logger()


def set_backend(backend):
    """Setup backend to the input one.

    Returns:
        the name of predefined backend

    Raises:
        if no valid backend setting is found in either environment variable or config file,
        stop the current run and print error message.

    """
    backend_name = backend
    if backend_name not in supported_backends():
        logger.warning(_BACKEND_NOT_FOUND)
        backend_name = _PYTORCH
    logger.debug("Using Backend: %s" % (backend_name))
    os.environ[_DGL_BACKEND] = backend_name
    global _BACKEND
    _BACKEND = backend_name
