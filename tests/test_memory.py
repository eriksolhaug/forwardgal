"""forwardgal.memory: a process's memory footprint (the GUI's memory limit)."""

import os
import sys

import numpy as np
import pytest

from forwardgal.memory import footprint


@pytest.mark.skipif(not (sys.platform == "darwin" or sys.platform.startswith("linux")), reason="macOS / Linux only")
def test_footprint_of_this_process_grows_with_an_array():
    before = footprint(os.getpid())
    a = np.ones(int(5e7))  # 400 MB, touched
    after = footprint(os.getpid())
    assert before > 10e6 and after - before > 300e6
    del a


def test_no_such_process():
    assert footprint(2**22 + 12345) is None
