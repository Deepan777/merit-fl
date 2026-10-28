"""Free commit memory, used by the run queues to start a worker only when enough memory is available."""
import os


def free_commit_gb():
    """Commit memory still available (RAM + page file) on Windows; available RAM elsewhere."""
    if os.name == "nt":
        import ctypes

        names = ("total_phys", "avail_phys", "total_pagefile", "avail_pagefile",
                 "total_virtual", "avail_virtual", "avail_extended_virtual")

        class MemStatus(ctypes.Structure):
            _fields_ = ([("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)]
                        + [(n, ctypes.c_ulonglong) for n in names])

        m = MemStatus()
        m.dwLength = ctypes.sizeof(MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.avail_pagefile / 2**30
    import psutil
    return psutil.virtual_memory().available / 2**30
