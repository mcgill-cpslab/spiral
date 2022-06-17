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
"""Common functions for the spiro which are used in multiple places."""


import numpy as np
from numpy import ndarray

from spiro.core.config import _BACKEND


def is_sorted(array_to_check: ndarray) -> bool:
    """Check if the input 1 d array is sorted asc.

    Args:
        array_to_check: an 1 D numpy array of ints

    Return:
        Boolean
    """
    return bool(np.all(array_to_check[:-1] <= array_to_check[1:]))


def get_anchor_position(arr_to_search: ndarray, anchors: ndarray) -> ndarray:
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


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    r"""Call in a loop to create terminal progress bar.

    Args:
        iteration: current iteration (Int)
        total: total iterations (Int)
        prefix: prefix string (Str)(Optional)
        suffix: suffix string (Str)(Optional)
        decimals: positive number of decimals in percent complete (Int)(Optional)
        length: character length of bar (Int)(Optional)
        fill: bar fill character (Str)(Optional)
        printEnd: end character (e.g. "\r", "\r\n") (Str)(Optional)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
