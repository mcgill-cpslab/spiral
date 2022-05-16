#!/usr/bin/env python
"""Tests `spiro.core.config` package."""

from spiro.core.dataio import neg_sampling
from tests.core.common_functions import *
import numpy as np
from spiro.core.logger import get_logger

logger = get_logger()

def test_negative_sampling():
    pos_id = np.array([3,5,6,8,9,22])
    sample = neg_sampling(pos_id, 25, 10)
    check = [i not in sample for i in pos_id]
    assert np.all(check)


