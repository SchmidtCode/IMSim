import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from imsim.wsgi import app, server

__all__ = ["app", "server"]

if __name__ == "__main__":
    from imsim.__main__ import main

    main()
