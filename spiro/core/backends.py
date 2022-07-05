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
"""Supporting backends.

This module lists the backends that we support.

Example:
    >>> from spiro.core import backends
    >>> print(backends.supported_backends())
"""

from typing import List

TENSORFLOW = "tensorflow"
PYTORCH = "pytorch"


def supported_backends() -> List[str]:
    """A function to return the list of backends that Nine Turn supports.

    Returns:
        a list of supported banckend names in string
    """
    return [TENSORFLOW, PYTORCH]
