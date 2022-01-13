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
"""Common functions for the nineturn which are used in multiple places."""


import numpy as np
from numpy import array

from nineturn.core.config import _BACKEND


def is_sorted(array_to_check: array) -> bool:
    """Check if the input 1 d array is sorted asc.

    Args:
        array_to_check: an 1 D numpy array of ints

    Return:
        Boolean
    """
    return np.all(array_to_check[:-1] <= array_to_check[1:])


def get_anchor_position(arr_to_search: array, anchors: array) -> array:
    """Get the position of anchors in a sorted array.

    Args:
        arr_to_search: an sorted numpy array
        anchors: an numpy array with unique values. Each values should present in the arr_to_search.

    Return:
        anchor's last position in arr_to_search. Position count start from 1

    Example:
        >>> a = np.array([0,0,1,1,2,3,3,3,4,5])
        >>> b = np.unique(a)
        >>> c = get_anchor_position(a,b)
        >>> c
        array([2, 4, 5, 8, 9, 10])
    """
    anchor_position = np.searchsorted(arr_to_search, anchors)[1:]
    last_anchor = arr_to_search.shape[0]
    return np.hstack((anchor_position, [last_anchor]))


def _get_backend() -> str:
    """Internal functions to return the current backend."""
    return _BACKEND
