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
import logging
import logging.config
import os
import sys
from typing import Optional

from nineturn.core.backends import PYTORCH, TENSORFLOW, supported_backends

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


class _ExcludeErrorsFilter(logging.Filter):
    def filter(self, record):
        """Only lets through log messages with log level below ERROR (numeric value: 40)."""
        return record.levelno < 40


format_string = "%(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d] %(message)s"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": "false",
    'filters': {'exclude_errors': {'()': _ExcludeErrorsFilter}},
    "formatters": {
        "basic": {
            "class": "logging.Formatter",
            "datefmt": "%Y-%m-%d:%H:%M:%S",
            "format": format_string,
        }
    },
    'handlers': {
        'console_stderr': {
            # Sends log messages with log level ERROR or higher to stderr
            'class': 'logging.StreamHandler',
            'level': 'ERROR',
            'formatter': 'basic',
            'stream': sys.stderr,
        },
        'console_stdout': {
            # Sends log messages with log level lower than ERROR to stdout
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'basic',
            'filters': ['exclude_errors'],
            'stream': sys.stdout,
        },
        'file': {
            # Sends all log messages to a file
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'basic',
            'filename': 'nineturn.log',
            'encoding': 'utf8',
        },
    },
    'root': {
        # In general, this should be kept at 'NOTSET'.
        # Otherwise it would interfere with the log levels set for each handler.
        'level': 'NOTSET',
        'handlers': ['console_stderr', 'console_stdout', 'file'],
    },
}


def get_logger():
    """Internal function, return a logger specific to nine turn."""
    logging.basicConfig()
    logging.config.dictConfig(LOGGING)
    return logging.getLogger()


logger = get_logger()


def set_backend(backend=None) -> str:
    """Setup backend.

    if backend is defined, and the value is either 'tensorflow' or 'pytorch',
    then set up 'DGLBACKEND' for DGL runtime to its value and return it
    Otherwise, check if config file exist in working directory, use the backend set up there.
    if no config file, no input backend, use  environment variable 'NINETURN_BACKEND',

    Returns:
        the name of predefined backend

    Raises:
        if no valid backend setting is found in either environment variable or config file,
        stop the current run and print error message.

    """
    backend_name = ""
    working_dir = os.getcwd()
    config_path = os.path.join(working_dir, 'config.json')
    if backend:
        backend_name = backend
    elif os.path.exists(config_path):
        with open(config_path, "r") as config_file:
            config_dict = json.load(config_file)
            backend_name = config_dict.get('backend', '').lower()
    elif _NINETURN_BACKEND in os.environ:
        backend_name = os.getenv(_NINETURN_BACKEND, "")
    else:
        logger.warning(_BACKEND_NOT_SET)
        backend_name = _PYTORCH

    if backend_name not in supported_backends():
        logger.warning(_BACKEND_NOT_FOUND)
        backend_name = _PYTORCH
    logger.info("Using Nine Turn Backend: %s" % (backend_name))
    os.environ[_DGL_BACKEND] = backend_name
    global _BACKEND
    _BACKEND = backend_name
    import dgl

    return backend_name


def get_backend() -> Optional[str]:
    """Return the current backend."""
    global _BACKEND
    return _BACKEND
