"""Linux child launcher: a cancelled DTR worker must not leave NPU work running."""

import ctypes
import os
import resource
import signal
import sys


def main():
    parent = int(sys.argv[1])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "Cannot attach native runner lifetime to its worker")
    if os.getppid() != parent:
        raise SystemExit(1)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.execv(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()
