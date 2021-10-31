"""Main module."""

from dgl.nn import GraphConv


def func_t(a: int) -> int:
    """
    This is a test function.

    Args:
      a: int

    Returns:
      number: int
    """
    a = GraphConv(2, 3)
    return 5
