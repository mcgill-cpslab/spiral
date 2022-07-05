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
"""Core data loading functions."""

import numpy as np
from numpy import ndarray

from spiro.core.logger import get_logger

logger = get_logger()

spirograph_dataset = {'aminer': "https://originalstatic.aminer.cn/misc/dblp.v13.7z"}
dataset_version = {'aminer': 'V1'}
raw_data_name = {'aminer': 'aminer.zip'}


def neg_sampling(pos_inds: ndarray, n_items: int, n_samp: int):
    """Pre-verified with binary search, `pos_inds` is assumed to be ordered."""
    raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    ss = np.searchsorted(pos_inds_adj, raw_samp, side='right')
    neg_inds = raw_samp + ss
    return neg_inds
