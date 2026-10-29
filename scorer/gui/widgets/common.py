"""Shared widget helpers: safe file dialogs, a progress log, message boxes, number fields."""
from __future__ import annotations

import os
import time

from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import QFileDialog, QLineEdit, QMessageBox, QPlainTextEdit

from scorer.gui.core import meta as M

SETTINGS = ("scorer", "gui")  # QSettings organisation / application


def settings() -> QSettings:
    """Return the GUI's QSettings store (remembered folders, last run folder)."""
    return QSettings(*SETTINGS)


def start_dir(meta: dict, key: str | None = None) -> str:
    """Return the folder a dialog opens in: the last one used for ``key``, else the project folder, else the cwd."""
    if key:
        last = settings().value(f"dirs/{key}", "", type=str)
        if last and os.path.isdir(last):
            return last
    d = M.project_dir(meta)
    return d if os.path.isdir(d) else os.getcwd()


def _remember(key: str | None, path: str) -> None:
    """Store the folder of ``path`` as the last one used for dialog ``key``."""
    if key and path:
        settings().setValue(f"dirs/{key}", path if os.path.isdir(path) else os.path.dirname(path))


def pick_dir(parent, title: str, meta: dict, key: str | None = None) -> str:
    """Ask for a folder; returns '' if cancelled. ``key`` remembers the folder between sessions."""
    path = QFileDialog.getExistingDirectory(parent, title, start_dir(meta, key), QFileDialog.ShowDirsOnly)
    _remember(key, path)
    return path


def pick_file(parent, title: str, meta: dict, filt: str, key: str | None = None) -> str:
    """Ask for an existing file matching ``filt``; returns '' if cancelled."""
    path, _ = QFileDialog.getOpenFileName(parent, title, start_dir(meta, key), filt)
    _remember(key, path)
    return path


def pick_save(parent, title: str, meta: dict, suggested: str, filt: str, key: str | None = None) -> str:
    """Ask for a file name to save to, starting from ``suggested``; returns '' if cancelled."""
    base = start_dir(meta, key)
    path, _ = QFileDialog.getSaveFileName(parent, title, os.path.join(base, suggested), filt)
    _remember(key, path)
    return path


def error(parent, title: str, text: str) -> None:
    """Show an error dialog; text after the first blank line goes into the 'Show details' section."""
    box = QMessageBox(QMessageBox.Critical, title, text.split("\n\n")[0], parent=parent)
    if "\n\n" in text:
        box.setDetailedText(text)
    box.exec_()


def confirm(parent, title: str, text: str) -> bool:
    """Ask a yes/no question; return True for yes."""
    return QMessageBox.question(parent, title, text, QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes


def info(parent, title: str, text: str) -> None:
    """Show an information dialog."""
    QMessageBox.information(parent, title, text)


def int_field(text: str, lo: int = 0, hi: int = 2_000_000_000) -> QLineEdit:
    """Return a line edit that only accepts whole numbers in ``[lo, hi]``."""
    f = QLineEdit(text)
    f.setValidator(QIntValidator(lo, hi))
    return f


def float_field(text: str, lo: float = 0.0, hi: float = 1e9, decimals: int = 4) -> QLineEdit:
    """Return a line edit that only accepts decimal numbers in ``[lo, hi]``."""
    f = QLineEdit(text)
    v = QDoubleValidator(lo, hi, decimals)
    v.setNotation(QDoubleValidator.StandardNotation)
    f.setValidator(v)
    return f


def read_int(field: QLineEdit, name: str) -> int:
    """Return the field's value as an int.

    Raises:
        ValueError: Naming the field (``name``) if the text is not a whole number.
    """
    try:
        return int(field.text().strip())
    except ValueError:
        raise ValueError(f"{name}: {field.text()!r} is not a whole number") from None


def read_float(field: QLineEdit, name: str) -> float:
    """Return the field's value as a float; a decimal comma is accepted.

    Raises:
        ValueError: Naming the field (``name``) if the text is not a number.
    """
    try:
        return float(field.text().strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"{name}: {field.text()!r} is not a number") from None


class LogPanel(QPlainTextEdit):
    """Read-only, time-stamped progress log shown under long-running actions (keeps the last ``max_lines``)."""

    def __init__(self, max_lines: int = 2000):
        """Create the panel."""
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_lines)
        self.setMinimumHeight(120)

    def add(self, text: str) -> None:
        """Append ``text`` with the current time."""
        self.appendPlainText(f"{time.strftime('%H:%M:%S')}  {text}")
