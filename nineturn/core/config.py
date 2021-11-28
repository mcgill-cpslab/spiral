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
"""global config for runtime.

This module define the runtime context which include backend when imported
To use Nine Turn, you should always import this module and then dgl

Example:
    >>> from nineturn.core import config
    >>> import dgl

"""
import json
import os
import sys

_NINETURN_BACKEND = "NINETURN_BACKEND"
_DGL_BACKEND = "DGLBACKEND"
_TENSORFLOW = "tensorflow"
_PYTORCH = "pytorch"
_BACKEND_NOT_FOUND = f"""
    Nine Turn backend is not selected or invalid. Assuming {_PYTORCH} for now.
    Please set up environment variable '{_NINETURN_BACKEND}' to one of '{_TENSORFLOW}'
    and '{_PYTORCH}' to select your backend.
    """


def _set_backend() -> str:
    """Setup backend.

    if environment variable '_NINETURN_BACKEND' is set,
    and the value is either 'tensorflow' or 'pytorch',
    then set up 'DGLBACKEND' for DGL runtime to its value and return it

    Returns:
        the name of predefined backend

    Raises:
        if no valid backend setting is found in either environment variable or config file,
        stop the current run and print error message.

    """
    backend_name = ""
    working_dir = os.getcwd()
    config_path = os.path.join(working_dir, 'config.json')
    backend_name = os.getenv(_NINETURN_BACKEND, "")

    if os.path.exists(config_path):
        with open(config_path, "r") as config_file:
            config_dict = json.load(config_file)
            backend_name = config_dict.get('backend', '').lower()

    if backend_name not in [_TENSORFLOW, _PYTORCH]:
        print(_BACKEND_NOT_FOUND, file=sys.stderr)
    os.environ[_DGL_BACKEND] = backend_name
    return backend_name


_BACKEND = _set_backend()


def get_backend() -> str:
    """Retrieve backend value.

    Returns:
        the name of backend defined as either env variable or in config_file.
    """
    return _BACKEND
