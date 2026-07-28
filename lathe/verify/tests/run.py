"""
Test runner. Works from any directory and needs no installed package.

    python lathe/verify/tests/run.py
    python lathe/verify/tests/run.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def main(argv=None) -> int:
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    argv = list(sys.argv[1:] if argv is None else argv)
    verbosity = 2 if ("-v" in argv or "--verbose" in argv) else 1
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=HERE, pattern="test_*.py", top_level_dir=ROOT)
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
