"""How much memory a process uses (for the GUI's memory limit on fits; no Qt, no extra packages).

``footprint(pid)`` returns bytes, or None where it cannot be measured:

  macOS   phys_footprint (proc_pid_rusage): what Activity Monitor calls "Memory".
          It counts compressed and swapped pages, which the resident size
          leaves out (a resident-size guard missed a 4.6 GB, mostly
          compressed test run on 2026-11-22).
  Linux   VmRSS + VmSwap from /proc/<pid>/status.
  other   None (the limit is then not enforced).
"""

from __future__ import annotations

import ctypes
import sys


class _RusageV0(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
        "wired_size", "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime")]


def footprint(pid):
    """Memory of process ``pid`` in bytes (see the module docstring), or None."""
    if sys.platform == "darwin":
        info = _RusageV0()
        libc = ctypes.CDLL("libc.dylib")
        if libc.proc_pid_rusage(int(pid), 0, ctypes.byref(info)) != 0:  # 0 = RUSAGE_INFO_V0
            return None
        return int(info.phys_footprint)
    if sys.platform.startswith("linux"):
        try:
            fields = dict(line.split(":", 1) for line in open(f"/proc/{int(pid)}/status"))
        except OSError:
            return None
        kb = sum(int(fields[k].split()[0]) for k in ("VmRSS", "VmSwap") if k in fields)
        return kb * 1024
    return None
