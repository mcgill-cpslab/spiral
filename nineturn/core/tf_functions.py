"""Tensorflow specific common functions."""
import tensorflow as tf
from numpy import ndarray
from typing import List

def _to_tensor(arr: ndarray) -> tf.Tensor:
    """Convert a numpy array to tensorflow tensor."""
    return tf.constant(arr)


def nt_layers_list() -> List:
    return []
