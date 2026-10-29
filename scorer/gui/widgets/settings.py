"""
Settings / metadata tab.

Changes: one form row per key that SHOWS the current value (before, the boxes stayed empty after
loading a JSON or pressing defaults, and each value needed its own "set" button); edits apply when
you leave the field; validation lists the actual problems instead of printing to the console and no
longer crashes on numeric values or rejects IDs like '4F'; 'reset' keeps project_path usable.
"""
from __future__ import annotations

import json

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
                             QVBoxLayout, QWidget)

from scorer.gui.core import meta as M
from scorer.gui.widgets import common as C

TOOLTIPS = {
    "sample_rate": "sample rate of the data on disk (raw file before preprocessing; updated after resampling)",
    "ecog_channels": "comma-separated channel indices, e.g. 0,1",
    "emg_channels": "comma-separated channel indices, e.g. 2",
    "time_channel": "index of the time column in the csv",
    "ylim": "infer | infer_ephys | standard",
    "device": "auto (GPU if available) | cuda | cpu",
}


class SettingsWidget(QWidget):
    """Settings tab: one form row per metadata key, plus save/load/validate buttons.

    Edits apply when you leave a field. The widget changes the shared metadata dict in
    place (never replaces it), so every tab always sees the same values.

    Signals:
        metadata_changed(): A value was edited, loaded or reset.

    Args:
        metadata: The shared metadata dict.
    """
    metadata_changed = pyqtSignal()

    def __init__(self, metadata: dict) -> None:
        """Build the form and buttons and show the current values."""
        super().__init__()
        self.params = metadata
        layout = QVBoxLayout(self)
        self.status = QLabel("")
        layout.addWidget(self.status)

        path_row = QHBoxLayout()
        self.path_label = QLabel()
        btn_path = QPushButton("select project folder")
        btn_path.clicked.connect(self.set_project_path)
        path_row.addWidget(QLabel("project folder:"))
        path_row.addWidget(self.path_label, 1)
        path_row.addWidget(btn_path)
        layout.addLayout(path_row)

        form = QFormLayout()
        self.fields: dict[str, QWidget] = {}
        for key in M.PARAM_KEYS:
            if key == "project_path":
                continue
            if key == "ylim":
                w = QComboBox()
                w.addItems(list(M.YLIM_MODES))
                w.setEditable(False)
                w.currentTextChanged.connect(lambda v, k=key: self._set(k, v))
            elif key == "device":
                w = QComboBox()
                w.addItems(list(M.DEVICES))
                w.currentTextChanged.connect(lambda v, k=key: self._set(k, v))
            else:
                w = QLineEdit()
                w.editingFinished.connect(lambda k=key, f=w: self._set(k, f.text()))
                if key == "scoring_started":
                    w.setReadOnly(True)
            w.setToolTip(TOOLTIPS.get(key, ""))
            self.fields[key] = w
            form.addRow(key, w)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        for text, slot in (("fill defaults", self.default_metadata), ("validate", lambda: self.validate_metadata()),
                           ("save to .json", self.save_metadata), ("load .json", self.load_metadata),
                           ("new session (reset)", self.reset_metadata)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        layout.addLayout(buttons)

        layout.addWidget(QLabel("all metadata (including values written by preprocessing and scoring):"))
        self.dump = QPlainTextEdit()
        self.dump.setReadOnly(True)
        layout.addWidget(self.dump, 1)
        self.refresh()

    # ---------------------------------------------------------------------------------------
    def refresh(self) -> None:
        """Show the dict's current values in every field (without triggering edits)."""
        for key, w in self.fields.items():
            v = self.params.get(key)
            text = "" if v is None else str(v)
            w.blockSignals(True)
            if isinstance(w, QComboBox):
                if text and w.findText(text) < 0:
                    w.addItem(text)
                w.setCurrentText(text or w.itemText(0))
            else:
                w.setText(text)
            w.blockSignals(False)
        self.path_label.setText(M.project_dir(self.params))
        self.dump.setPlainText(json.dumps(self.params, indent=1, default=str))

    def _set(self, key: str, value: str) -> None:
        """Store a typed value (tidied by ``meta.normalise_value``) and announce the change."""
        value = M.normalise_value(key, value)
        if self.params.get(key) == value:
            return
        self.params[key] = value
        self.refresh()  # shows the tidied value (e.g. '250.0' -> '250')
        self.metadata_changed.emit()

    def set_project_path(self) -> None:
        """Ask for the project folder and store it."""
        path = C.pick_dir(self, "select project folder", self.params, "project")
        if path:
            self.params["project_path"] = path
            self.refresh()
            self.metadata_changed.emit()

    def default_metadata(self) -> None:
        """Fill empty fields with ``meta.DEFAULTS``; filled fields are kept."""
        for k, v in M.DEFAULTS.items():
            if self.params.get(k) in (None, ""):
                self.params[k] = v
        self.refresh()
        self.status.setText("defaults filled in for empty fields")
        self.metadata_changed.emit()

    def save_metadata(self) -> None:
        """Save the metadata to a JSON file chosen by the user and list values still empty."""
        from scorer.data.storage import save_metadata

        name = f"{M.get_str(self.params, 'scoring_started', M.timestamp())}_meta.json"
        path = C.pick_save(self, "save metadata", self.params, name, "JSON (*.json *.txt)", "metadata")
        if not path:
            return
        self.params["metadata_path"] = path
        save_metadata(path, self.params)
        self.status.setText(f"saved {path}")
        self.refresh()
        unset = [p.key for p in M.validate(self.params) if p.severity == "warning"]
        if unset:
            C.info(self, "metadata saved", "saved; these values are still empty:\n" + ", ".join(unset))

    def load_metadata(self) -> None:
        """Load metadata from a JSON file into the shared dict and validate it."""
        from scorer.data.storage import load_metadata

        path = C.pick_file(self, "load metadata", self.params, "JSON (*.json *.txt)", "metadata")
        if not path:
            return
        data = load_metadata(path)
        if not isinstance(data, dict):
            C.error(self, "load metadata", f"{path} does not contain a metadata dictionary")
            return
        self.params.clear()  # same dict object: every tab keeps seeing the shared metadata
        self.params.update(M.new_metadata())
        self.params.update(data)
        self.refresh()
        self.status.setText(f"loaded {path}")
        self.metadata_changed.emit()
        self.validate_metadata(quiet_if_ok=True)

    def reset_metadata(self) -> None:
        """After confirmation, start a new session: clear everything except the project folder."""
        if not C.confirm(self, "new session", "clear all metadata and start a new scoring session?"):
            return
        project = self.params.get("project_path")
        self.params.clear()
        self.params.update(M.new_metadata())
        if project:
            self.params["project_path"] = project
        self.refresh()
        self.metadata_changed.emit()

    def validate_metadata(self, quiet_if_ok: bool = False) -> bool:
        """Show the problems ``meta.validate`` finds.

        Args:
            quiet_if_ok: Show nothing when there is no problem.

        Returns:
            True if there are no errors (warnings are allowed).
        """
        problems = M.validate(self.params)
        errors = [p for p in problems if p.severity == "error"]
        if not problems and quiet_if_ok:
            return True
        text = "All parameters look fine." if not problems else "\n".join(map(str, problems))
        (C.error if errors else C.info)(self, "parameter check", text)
        return not errors
