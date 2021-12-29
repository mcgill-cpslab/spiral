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
"""Common functions for the nineturn which are used in multiple places."""


import numpy as np
from numpy import array


def is_sorted(array_to_check: array) -> bool:
    """Check if the input 1 d array is sorted asc.

    Args:
        array_to_check: an 1 D numpy array of ints

    Return:
        Boolean
    """
    return np.all(array_to_check[:-1] <= array_to_check[1:])
