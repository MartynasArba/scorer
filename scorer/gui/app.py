"""
Entry point:  python -m scorer.gui   (or python -m scorer.gui.app)

Replaces gui/main.py, which started the application on import, had no error handling (since
PyQt 5.5 an exception escaping a slot aborts the whole program, losing unsaved labels) and set
torch.set_grad_enabled(False) only for the main thread (scoring now uses torch.inference_mode).
"""
from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path


def _install_excepthook(app) -> None:
    """Show uncaught exceptions in a dialog and the log file instead of letting them end the app.

    Since PyQt 5.5 an exception escaping a slot aborts the program, which would lose
    unsaved labels.
    """
    from PyQt5.QtWidgets import QMessageBox

    def hook(exc_type, exc, tb):
        """Log the exception with its traceback and show it in an error dialog."""
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logging.getLogger("scorer.gui").error("unhandled exception\n%s", text)
        box = QMessageBox(QMessageBox.Critical, "error", f"{exc_type.__name__}: {exc}")
        box.setInformativeText("The GUI keeps running; details are in the log file.")
        box.setDetailedText(text)
        box.exec_()

    sys.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    """Start the GUI and return Qt's exit code.

    Logs go to ``~/.scorer_gui.log`` and the console. High-DPI scaling is enabled.

    Args:
        argv: Arguments for QApplication; None uses ``sys.argv``.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    log_file = Path.home() / ".scorer_gui.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()])
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv if argv is None else argv)
    _install_excepthook(app)

    from scorer.gui.main_window import SleepWindow

    win = SleepWindow()
    win.show()
    logging.getLogger("scorer.gui").info("GUI started; log file %s", log_file)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
