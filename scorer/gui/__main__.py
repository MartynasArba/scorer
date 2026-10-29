"""Entry point for ``python -m scorer.gui``."""
import sys

from scorer.gui.app import main

if __name__ == "__main__":  # guard: spawned worker processes (Windows) re-import this module
    sys.exit(main())
