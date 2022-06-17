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


class DimensionError(Exception):
    """Nine Turn error specific to incorrect input dimension."""

    def __init__(self, message):
        """Just need the error message to tell what is the error."""
        super().__init__(message)


class ValueNotSortedError(Exception):
    """Nine Turn error specific to not sorted inputs."""

    def __init__(self, message):
        """Just need the error message to tell what is the error."""
        super().__init__(message)


class BackendNotSupportedError(Exception):
    """Nine Turn error specific to not supported backends."""

    def __init__(self, message):
        """Just need the error message to tell what is the error."""
        super().__init__(message)


class ValueError(Exception):
    """Nine Turn error specific to not sorted inputs."""

    def __init__(self, message):
        """Just need the error message to tell what is the error."""
        super().__init__(message)
