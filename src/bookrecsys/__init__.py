"""BookRecSys: Personalized local-first + online book & light/web novel recommender."""

import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

__version__ = "0.1.0"
