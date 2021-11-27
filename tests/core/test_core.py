#!/usr/bin/env python
"""Tests `nineturn.core` package."""

from nineturn.core.utils import func_t


def test_all():
    """Test that __all__ contains only names that are actually exported."""
    import nineturn.core as core

    missing = set(n for n in core.__all__ if getattr(core, n, None) is None)
    assert len(missing) == 0, "__all__ contains unresolved names: %s" % (", ".join(missing),)


def test_func():
    """Test the dummy example function."""
    assert func_t(5) == 5
