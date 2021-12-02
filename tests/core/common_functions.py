#!/usr/bin/env python
# flake8: noqa
"""Tests `nineturn.core.config` package."""

import os
import sys

dummy = "dummy"
NINETURN_BACKEND = "NINETURN_BACKEND"
DGL_BACKEND = "DGLBACKEND"
TENSORFLOW = "tensorflow"
PYTORCH = "pytorch"
BACKEND_NOT_FOUND = f"""
    Nine Turn backend is not selected or invalid. Assuming {PYTORCH} for now.
    Please set up environment variable '{NINETURN_BACKEND}' to one of '{TENSORFLOW}'
    and '{PYTORCH}' to select your backend.
    """


def clear_background():
    modules_to_clear = [k for k in sys.modules.keys() if 'nineturn' in k]
    for k in modules_to_clear:
        del sys.modules[k]
    if DGL_BACKEND in os.environ:
        del os.environ[DGL_BACKEND]
    if NINETURN_BACKEND in os.environ:
        del os.environ[NINETURN_BACKEND]


def set_background():
    clear_background()
    os.environ[NINETURN_BACKEND] = TENSORFLOW


def set_invalid_background():
    clear_background()
    os.environ[NINETURN_BACKEND] = dummy
