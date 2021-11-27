#!/usr/bin/env python
"""Tests `nineturn.core` package."""


def test_all():
    """Test that __all__ contains only names that are actually exported."""
    from nineturn import core

    missing = set(n for n in core.__all__ if getattr(core, n, None) is None)
    assert (len(missing) == 0, "__all__ contains unresolved names: %s" % (", ".join(missing),))
