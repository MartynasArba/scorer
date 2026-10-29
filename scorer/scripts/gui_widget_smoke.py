"""
Widget-logic smoke test WITHOUT a display or PyQt5: a permissive stand-in for PyQt5 is put on the
path so the tab classes can be built and driven (load data, label with the mouse, undo, save,
autosave, load autoscored states with confidence, navigate, score with a fake model, compare).
It checks the GUI's own logic, not Qt rendering - run the real GUI for that.

    python -m scorer.scripts.gui_widget_smoke
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------------------------
# minimal PyQt5 stand-in
# ---------------------------------------------------------------------------------------------
class _Sig:
    """Stand-in for a bound Qt signal: stores connected slots and calls them on ``emit``."""
    def __init__(self):
        self.slots = []

    def connect(self, fn):
        self.slots.append(fn)

    def emit(self, *a):
        import inspect

        for fn in list(self.slots):
            try:  # like PyQt: a slot that takes fewer arguments gets only the first ones
                ps = inspect.signature(fn).parameters.values()
                if not any(p.kind == p.VAR_POSITIONAL for p in ps):
                    n = sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
                            for p in ps)
                    a = a[:max(n, 0)] if n < len(a) else a
            except (TypeError, ValueError):
                pass
            fn(*a)

    def __getitem__(self, _):  # buttonClicked[int]
        return self


class pyqtSignal:  # noqa: N801
    """Stand-in for ``pyqtSignal``: gives each widget instance its own ``_Sig``."""
    def __init__(self, *types_):
        self.name = None

    def __set_name__(self, owner, name):
        self.name = "_sig_" + name

    def __get__(self, obj, owner):
        if obj is None:
            return self
        if self.name not in obj.__dict__:
            obj.__dict__[self.name] = _Sig()
        return obj.__dict__[self.name]


class _StubMeta(type):
    """Metaclass that answers any class attribute (e.g. ``QSizePolicy.Expanding``) with a stub."""
    def __getattr__(cls, name):  # class constants such as QSizePolicy.Expanding
        return 0


class Stub(metaclass=_StubMeta):
    """accepts any constructor / method call; unknown attributes are no-op callables or signals"""
    _signals = ("clicked", "toggled", "textChanged", "editingFinished", "currentTextChanged", "valueChanged",
                "currentIndexChanged", "stateChanged", "timeout", "started", "finished", "buttonClicked", "activated",
                "modificationChanged")

    def __init__(self, *a, **k):
        self._text = a[0] if a and isinstance(a[0], str) else ""
        self._checked = False
        self._value = 0
        self._items: list[str] = []
        self._index = -1
        self._enabled = True
        self._buttons = {}

    def __getattr__(self, name):
        if name in Stub._signals:
            s = _Sig()
            self.__dict__[name] = s
            return s
        return lambda *a, **k: Stub()

    # text
    def setText(self, t): self._text = str(t)  # noqa: E704
    def text(self): return self._text  # noqa: E704
    def setPlainText(self, t): self._text = t  # noqa: E704
    def appendPlainText(self, t): self._text += t + "\n"  # noqa: E704
    def toPlainText(self): return self._text  # noqa: E704

    # checkable
    def setChecked(self, v):
        changed = bool(v) != self._checked
        self._checked = bool(v)
        if changed and "toggled" in self.__dict__:
            self.toggled.emit(self._checked)

    def isChecked(self): return self._checked  # noqa: E704

    # value widgets
    def setValue(self, v): self._value = int(v)  # noqa: E704
    def value(self): return self._value  # noqa: E704
    def setRange(self, lo, hi): self._value = min(max(self._value, lo), hi)  # noqa: E704

    # combo boxes
    def addItems(self, items): self._items += list(items); self._index = 0 if self._index < 0 else self._index  # noqa
    def addItem(self, it): self._items.append(it); self._index = 0 if self._index < 0 else self._index  # noqa
    def clear(self): self._items, self._index = [], -1  # noqa: E704
    def count(self): return len(self._items)  # noqa: E704
    def itemText(self, i): return self._items[i] if 0 <= i < len(self._items) else ""  # noqa: E704
    def findText(self, t): return self._items.index(t) if t in self._items else -1  # noqa: E704
    def currentText(self): return self.itemText(self._index)  # noqa: E704
    def currentIndex(self): return self._index  # noqa: E704
    def setCurrentText(self, t): self._index = self.findText(t) if t in self._items else self._index  # noqa: E704

    def setCurrentIndex(self, i):
        self._index = i
        if "currentIndexChanged" in self.__dict__:
            self.currentIndexChanged.emit(i)

    def setEnabled(self, v): self._enabled = bool(v)  # noqa: E704
    def isEnabled(self): return self._enabled  # noqa: E704

    # button group
    def addButton(self, b, i): self._buttons[i] = b  # noqa: E704
    def button(self, i): return self._buttons.get(i)  # noqa: E704


class QListWidgetItem(Stub):
    """List item stand-in that keeps its data and check state."""
    def __init__(self, text=""):
        super().__init__(text)
        self._data = None
        self._state = 2

    def setData(self, role, v): self._data = v  # noqa: E704
    def data(self, role): return self._data  # noqa: E704
    def flags(self): return 0  # noqa: E704
    def setCheckState(self, s): self._state = s  # noqa: E704
    def checkState(self): return self._state  # noqa: E704


class QListWidget(Stub):
    """List widget stand-in backed by a Python list."""
    def __init__(self, *a):
        super().__init__()
        self.list = []

    def addItem(self, it): self.list.append(it)  # noqa: E704
    def count(self): return len(self.list)  # noqa: E704
    def item(self, i): return self.list[i]  # noqa: E704
    def currentItem(self): return self.list[0] if self.list else None  # noqa: E704
    def clear(self): self.list = []  # noqa: E704


class QSettings(Stub):
    """QSettings stand-in backed by a class-level dict."""
    store: dict = {}

    def value(self, k, default=None, type=None): return QSettings.store.get(k, default)  # noqa: E704
    def setValue(self, k, v): QSettings.store[k] = v  # noqa: E704


class QMessageBox(Stub):
    """Message-box stand-in that records what was shown and answers questions with Yes."""
    Yes, No, Save, Discard, Cancel, Critical, Information = 1, 2, 3, 4, 5, 6, 7
    answers: list = []
    shown: list = []

    @staticmethod
    def question(*a, **k):
        QMessageBox.shown.append(a[2] if len(a) > 2 else "")
        return QMessageBox.answers.pop(0) if QMessageBox.answers else QMessageBox.Yes

    @staticmethod
    def information(*a, **k): QMessageBox.shown.append(a[2] if len(a) > 2 else "")  # noqa: E704
    def exec_(self): QMessageBox.shown.append(self._text)  # noqa: E704


class _Doc:
    """Text-document stand-in with a modified flag."""
    def __init__(self):
        self.modified = False

    def isModified(self): return self.modified  # noqa: E704
    def setModified(self, v): self.modified = bool(v)  # noqa: E704


class QPlainTextEdit(Stub):
    """Text-editor stand-in with a document."""
    def __init__(self, *a):
        super().__init__()
        self._doc = _Doc()

    def document(self): return self._doc  # noqa: E704

    def setPlainText(self, t):
        self._text = t
        self._doc.modified = False


class QProcess(Stub):
    """QProcess stand-in: records the command and lets the test feed output and finish it."""
    NotRunning, Running, MergedChannels = 0, 2, 1
    started: list = []

    def __init__(self, *a):
        super().__init__()
        self._state, self.out = 0, b""
        self.readyReadStandardOutput, self.finished, self.errorOccurred = _Sig(), _Sig(), _Sig()

    def start(self, program, args):
        QProcess.started.append([program] + list(args))
        self._state = 2

    def state(self): return self._state  # noqa: E704

    def readAllStandardOutput(self):
        out, self.out = self.out, b""
        return out

    def finish(self, code):
        self._state = 0
        self.finished.emit(code, 0)


class _Qt:
    """Stand-in for the ``Qt`` namespace (every constant is 0 unless set)."""
    Checked, Unchecked, UserRole = 2, 0, 256

    def __getattr__(self, name): return 0  # noqa: E704


def install_fake_qt() -> None:
    """Put the PyQt5 and matplotlib-Qt stand-ins into ``sys.modules``."""
    qtw = types.ModuleType("PyQt5.QtWidgets")
    for name in ("QWidget", "QMainWindow", "QTabWidget", "QVBoxLayout", "QHBoxLayout", "QGridLayout", "QFormLayout",
                 "QGroupBox", "QLabel", "QLineEdit", "QPushButton", "QCheckBox", "QComboBox", "QSpinBox", "QSlider",
                 "QRadioButton", "QButtonGroup", "QPlainTextEdit", "QTextEdit", "QDialog", "QInputDialog",
                 "QFileDialog", "QSizePolicy", "QShortcut", "QApplication", "QSplitter", "QTableWidget",
                 "QTableWidgetItem", "QHeaderView", "QProgressBar"):
        setattr(qtw, name, type(name, (Stub,), {}))
    qtw.QListWidget, qtw.QListWidgetItem, qtw.QMessageBox = QListWidget, QListWidgetItem, QMessageBox
    qtw.QPlainTextEdit = QPlainTextEdit
    qtc = types.ModuleType("PyQt5.QtCore")
    qtc.pyqtSignal, qtc.Qt, qtc.QSettings = pyqtSignal, _Qt(), QSettings
    for name in ("QObject", "QThread", "QTimer"):
        setattr(qtc, name, type(name, (Stub,), {}))
    qtc.QFileInfo = lambda p: types.SimpleNamespace(baseName=lambda: Path(p).name.split(".")[0])
    env_cls = type("QProcessEnvironment", (Stub,), {"systemEnvironment": staticmethod(lambda: Stub())})
    qtc.QProcess, qtc.QProcessEnvironment, qtc.QUrl = QProcess, env_cls, Stub
    qtg = types.ModuleType("PyQt5.QtGui")
    for name in ("QKeySequence", "QIntValidator", "QDoubleValidator", "QFont", "QDesktopServices"):
        setattr(qtg, name, type(name, (Stub,), {"StandardNotation": 0}))
    pkg = types.ModuleType("PyQt5")
    pkg.QtWidgets, pkg.QtCore, pkg.QtGui = qtw, qtc, qtg
    sys.modules.update({"PyQt5": pkg, "PyQt5.QtWidgets": qtw, "PyQt5.QtCore": qtc, "PyQt5.QtGui": qtg})

    class Canvas(Stub):
        def __init__(self, fig):
            super().__init__()
            self.figure = fig
            self.n = 0

        def mpl_connect(self, ev, fn):
            self.n += 1
            return self.n

        def draw_idle(self): self.figure.canvas.draw()  # noqa: E704  (real Agg draw: catches artist errors)

    mod = types.ModuleType("matplotlib.backends.backend_qt5agg")
    mod.FigureCanvasQTAgg = Canvas
    sys.modules["matplotlib.backends.backend_qt5agg"] = mod


def install_fake_scorer() -> None:
    """stand-in for inference.scoring (needs torch): classifies by window variance"""
    mod = types.ModuleType("scorer.inference.scoring")

    class Res:
        def __init__(self, codes, conf):
            self.gui_codes, self.confidence, self.classes = codes, conf, codes

    class FakeScorer:
        epoch_seconds = 4.0

        def __init__(self, enc, head, device="auto"):
            self.enc, self.head = enc, head

        def describe(self):
            return {"head": "context_mlp", "encoder_stage": "supcon", "mode": "zeroshot", "classes": ["W", "N", "R"],
                    "sample_rate": 128.0, "win_len": 512, "trained_on": {"train": ["mssv"]}}

        def score_windows(self, x, smooth_sigma=0.0, heuristics="none", source_sample_rate=None):
            assert source_sample_rate == 250.0, source_sample_rate
            v = x[:, 0].std(-1)
            codes = np.where(v > np.median(v), 1, 2)
            return Res(codes, np.clip(v / v.max(), 0, 1).astype(np.float32))

    def save_states(res, path):
        with open(path, "wb") as f:
            pickle.dump(res.gui_codes, f)
        with open(Path(path).with_name(Path(path).stem + "_confidence.pkl"), "wb") as f:
            pickle.dump(res.confidence, f)

    mod.SleepScorer, mod.save_states = FakeScorer, save_states
    sys.modules["scorer.inference.scoring"] = mod


def sync_run_task(parent, fn, *args, on_done=None, on_error=None, on_progress=None, on_finished=None, **kw):
    """Synchronous stand-in for ``workers.run_task``: runs the job at once and calls the callbacks."""
    import inspect

    params = inspect.signature(fn).parameters
    if "progress" in params:
        kw["progress"] = on_progress or (lambda s: None)
    if "cancelled" in params:
        kw["cancelled"] = lambda: False
    try:
        r = fn(*args, **kw)
        on_done and on_done(r)
    except Exception as e:
        if on_error is None:
            raise
        on_error(str(e))
    on_finished and on_finished()
    return types.SimpleNamespace(running=False, cancel=lambda: None)


def main() -> None:
    """Build every tab with the stand-ins and drive the main user paths; raises on the first failure."""
    import matplotlib

    matplotlib.use("Agg")
    tmp = Path(tempfile.mkdtemp(prefix="gui_widgets_"))
    try:
        install_fake_qt()
        scores_dir = tmp / "scores"  # storage.construct_paths: <data>/../../scores
        install_fake_scorer()
        from scorer.gui.widgets import autoscoring as AS
        from scorer.gui.widgets import common as C
        from scorer.gui import main_window as MW

        AS.run_task = sync_run_task
        errors = []
        C.error = lambda parent, title, text: errors.append((title, text))

        # data: one recording, GUI layout [C, N, T], 250 Hz, 4-s windows
        rec = tmp / "data" / "windowed_rec1"
        rec.mkdir(parents=True)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 300, 1000)).astype(np.float32)
        x[:, 100:200] *= 4
        np.save(rec / "X_chunk0.npy", x)

        win = MW.SleepWindow()
        meta = win.metadata
        meta.update(sample_rate="250", ecog_channels="0", emg_channels="1", scorer="MA", ylim="infer",
                    rec_start="2025-11-27 23:58:00", project_path=str(tmp))
        win.settings_tab.refresh()
        assert win.settings_tab.fields["sample_rate"].text() == "250"

        # ---- automatic scoring: run folder -> model pair -> score -> open in labeling -------------
        run = tmp / "run"
        run.mkdir()
        for f in ("enc.pt", "head.pt"):
            (run / f).write_bytes(b"x")
        (run / "manifest.json").write_text(json.dumps({"stages": {
            "embed@zeroshot": {"encoder_path": str(run / "enc.pt"), "encoder_stage": "supcon"},
            "context_mlp@zeroshot": {"head": str(run / "head.pt")}}}))
        a = win.auto_scoring_tab
        a._load_run(str(run))
        assert a.choice is not None and a.choice.mode == "zeroshot"
        C.pick_dir = lambda *args, **k: str(tmp / "data")
        a.add_data()
        assert a.targets.count() == 1
        a.out_dir = str(tmp / "scores")
        a.run_scoring()
        assert not errors, errors
        assert a.results and a.results[0].ok, [o.message for o in a.results]
        states_path = a.results[0].states_path
        assert Path(states_path).exists() and Path(states_path).with_name(
            Path(states_path).stem + "_scoring.json").exists()
        assert meta["autoscoring"]["model"]["head"] == "context_mlp"

        a._open_result()  # -> labeling tab loads folder + states + confidence
        lab = win.scoring_tab
        assert lab.n == 300 and lab.track.confidence is not None and not lab.track.dirty, (lab.n, errors)
        assert set(np.unique(lab.states)) <= {1, 2}

        # ---- labeling with the mouse on the state axis (or its confidence twin) -------------------
        lab.increase_scale()
        lab.increase_scale()  # 3 windows visible
        ev = lambda xd, ax=None: types.SimpleNamespace(button=1, inaxes=ax or lab.view.conf_ax, xdata=xd)  # noqa
        lab.labeling_group.buttonClicked.emit(4)
        lab._on_press(ev(0.5))
        lab._on_move(ev(6.0))
        lab._on_release(ev(9.0))
        assert lab.states[0:3].tolist() == [4, 4, 4] and lab.track.dirty
        lab.undo()
        assert lab.states[0] in (1, 2)
        lab._on_press(ev(1.0))
        lab._on_release(ev(1.5))
        assert lab.states[0] == 4

        # ---- navigation ------------------------------------------------------------------------------
        lab.next_change()
        assert lab.current_idx == lab.track.next_change(0) or lab.current_idx > 0
        lab.conf_thr.setText("0.3")
        before = lab.current_idx
        lab.next_uncertain()
        assert lab.current_idx > before
        sys.modules["PyQt5.QtWidgets"].QInputDialog.getText = staticmethod(lambda *a, **k: ("00:02:00", True))
        lab.jump()  # recording starts 23:58:00 -> 00:02:00 is 240 s later (across midnight) -> window 60
        assert lab.current_idx == 60, lab.current_idx
        lab.toggle_macro_view()
        assert lab.scale == 150
        lab.next()
        assert lab.current_idx == 160
        lab.toggle_macro_view()
        lab.spec_check.setChecked(True)  # rebuild with spectrogram panel
        assert lab.view.spec_ax is not None
        lab.set_yscale(2.0)
        lab.ephys_only.setChecked(True)

        # ---- save / autosave / close guard -----------------------------------------------------------
        lab.track.set_range(5, 6, 0)
        lab.autosave()
        lab.track.set_range(7, 8, 0)
        lab.autosave()  # a second autosave reuses the same file
        autosaves = [p for p in scores_dir.iterdir() if "autosave" in p.name]
        assert len(autosaves) == 1, autosaves
        assert lab.has_unsaved()
        lab.save_states()
        assert not lab.has_unsaved() and any("frame" in p.name and "autosave" not in p.name for p in scores_dir.iterdir())

        # ---- report tab: epoch seconds -> window samples; agreement over real codes -----------------
        r = win.report_tab
        r.states_path = states_path
        assert r._win_len() == 1000
        r.get_hypnogram()
        r.get_state_report()
        manual = tmp / "manual.pkl"
        with open(manual, "wb") as f:
            pickle.dump(lab.states, f)
        picks = iter([str(manual), states_path])
        C.pick_file = lambda *args, **k: next(picks)
        r._popup = lambda *args, **k: None
        r.compare_scorers()

        # ---- settings validation and reset keep a usable project path --------------------------------
        win.settings_tab.validate_metadata()
        win.settings_tab.reset_metadata()
        assert meta["project_path"] == str(tmp) and meta["sample_rate"] is None

        # ---- preprocessing option collection / checks (no file -> clear error, nothing runs) ----------
        p = win.preprocess_tab
        p.apply_model_preset()
        assert p.options().new_sr == 128 and p.options().win_len == 512
        p.run_preprocessing()
        assert errors and errors[-1][0] == "preprocessing", errors
        # ---- model training tab: config check -> command -> progress -> results -> scoring handoff ----
        n_err = len(errors)
        (tmp / "dsA").mkdir()
        cfg = tmp / "train.toml"
        cfg.write_text(f"""
[run]
name = "gui_run"
runs_dir = '{(tmp / "runs").as_posix()}'
[cache]
root = '{(tmp / "cache").as_posix()}'
[datasets]
a = '{(tmp / "dsA").as_posix()}'
b = '{(tmp / "data").as_posix()}'
[simclr]
datasets = ["a"]
[supcon]
datasets = ["a"]
[adversarial]
source = ["a"]
target = ["b"]
[context_rf]
train = ["a"]
[context_mlp]
train = ["a"]
val = ["b"]
[evaluate]
datasets = ["b"]
""")
        t = win.training_tab
        t.load_config(str(cfg))
        assert t.check is not None and t.check.ok, t.check_box.text()
        assert t.selected_stages() == ["inspect", "prepare", "simclr", "supcon", "adversarial", "embed", "context_rf",
                                       "context_mlp", "evaluate"], t.selected_stages()
        t.mode.setCurrentText("zeroshot")
        t.run_selected()  # adversarial still ticked in zeroshot mode -> refused
        assert len(errors) == n_err + 1 and "zeroshot" in errors[-1][1], errors[-1]
        t.run_name.setText("gui run 2")
        t.run_stages(["inspect"])
        cmd = QProcess.started[-1]
        assert cmd[1:4] == ["-u", "-m", "scorer.pipeline.run"] and "--stages" in cmd and 'run.name="gui run 2"' in cmd, cmd
        assert t.is_running() and win.auto_scoring_tab.gpu_busy()
        run_dir = tmp / "runs" / "gui run 2"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps({"stages": {
            "inspect": {"report": "x.json", "finished": "now"},
            "embed@zeroshot": {"encoder_path": str(run / "enc.pt"), "encoder_stage": "supcon", "finished": "now"},
            "context_mlp@zeroshot": {"head": str(run / "head.pt"), "finished": "now"}}}))
        (run_dir / "evaluate" / "zeroshot").mkdir(parents=True)
        (run_dir / "evaluate" / "zeroshot" / "summary.csv").write_text(
            "mode,head,dataset,accuracy,macro_f1,kappa\nzeroshot,context_mlp,b,0.91,0.88,0.85\n")
        t.proc.out = (b"2026 [INFO] scorer: ==================== stage inspect ====================\n"
                      b"INFO scorer: stage inspect done in 1s\nINFO scorer: all stages ")
        t._read_output()
        assert t.progress.done == ["inspect"] and "inspect" in t.status.text()
        t.proc.out = b"finished: ['inspect']\n"
        t._read_output()
        t.proc.finish(0)
        assert t.progress.finished and not t.is_running() and len(errors) == n_err + 1
        t._use_for_scoring()  # -> automatic scoring tab lists this run's heads
        assert a.choice is not None and a.choice.run == "gui run 2", a.choice
        print("[gui widgets] all checks passed;", len(errors), "expected error dialog(s)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
