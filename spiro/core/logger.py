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
"""The module is only used for logging."""

import logging
import logging.config
import sys


class _ExcludeErrorsFilter(logging.Filter):
    def filter(self, record):
        """Only lets through log messages with log level below ERROR (numeric value: 40)."""
        return record.levelno < 40


format_string = "%(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d] %(message)s"


def get_logger(log_file: str = 'spiro.log', level_to_file: str = 'INFO'):
    """Return the spiro logger. Not for used by library users."""
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
                'level': 'DEBUG',
                'formatter': 'basic',
                'filters': ['exclude_errors'],
                'stream': sys.stdout,
            },
            'file': {
                # Sends all log messages to a file
                'class': 'logging.FileHandler',
                'level': level_to_file,
                'formatter': 'basic',
                'filename': log_file,
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

    """Internal function, return a logger specific to nine turn."""
    logging.basicConfig()
    logging.config.dictConfig(LOGGING)
    return logging.getLogger()
