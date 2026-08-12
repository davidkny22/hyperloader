"""Host resource observations used by sizing formulae."""

from __future__ import annotations

import ctypes
import os
import sys


def free_host_memory() -> int:
    """Return currently available physical host memory in bytes."""
    if sys.platform == "win32":
        return _windows_free_memory()
    if sys.platform == "darwin":
        return _macos_free_memory()
    page_size = os.sysconf("SC_PAGE_SIZE")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    return int(page_size * available_pages)


def _macos_free_memory() -> int:
    """Return free and reclaimable inactive bytes from Mach host statistics."""
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    library.mach_host_self.restype = ctypes.c_uint
    library.host_statistics64.argtypes = (
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint),
    )
    library.host_statistics64.restype = ctypes.c_int

    host = library.mach_host_self()
    statistics = (ctypes.c_int * 64)()
    count = ctypes.c_uint(len(statistics))
    host_vm_info64 = 4
    result = library.host_statistics64(host, host_vm_info64, statistics, count)
    if result != 0:
        raise OSError(f"host_statistics64 failed with Mach status {result}")

    free_pages = ctypes.c_uint.from_buffer(statistics, 0).value
    inactive_pages = ctypes.c_uint.from_buffer(
        statistics, 2 * ctypes.sizeof(ctypes.c_uint)
    ).value
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    return page_size * (free_pages + inactive_pages)


def _windows_free_memory() -> int:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.available_physical)
