import tensorflow as tf
from numpy import ndarray


def to_tensor(arr: ndarray) -> tf.Tensor:
    """Convert a numpy array to tensorflow tensor."""
    return tf.constant(arr)
