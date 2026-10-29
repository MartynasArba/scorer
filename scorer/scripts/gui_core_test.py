"""
Checks for the Qt-free GUI core (no PyQt5, no torch needed):

    python -m scorer.scripts.gui_core_test
"""
from __future__ import annotations

import json
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    """Run all checks; raises AssertionError on the first failure, prints a line when all pass."""
    from scorer.gui.core import meta as M
    from scorer.gui.core.agreement import compare_states, format_agreement
    from scorer.gui.core.annotations import StateTrack, fit_array, fit_states_to, load_confidence_for, load_states_file
    from scorer.gui.core.autoscore import discover_models, expand_targets, output_path, source_rate_for
    from scorer.gui.core.plots import SignalView, TimeOfDayFormatter, confusion_figure, hypnogram, parse_rec_start
    from scorer.gui.core.preprocess_job import PreprocessOptions, check_options, model_preset
    from scorer.gui.core.recording import RecordingData, compute_ylims

    tmp = Path(tempfile.mkdtemp(prefix="gui_core_"))
    try:
        # ---- metadata ------------------------------------------------------------------------
        m = M.new_metadata()
        assert M.project_dir(m) == "." and M.get_float(m, "sample_rate") is None
        m.update(sample_rate=128, ecog_channels="0, 1", emg_channels="2", animal_id="4F", ylim="infer", device="cpu")
        assert M.parse_channels(m["ecog_channels"]) == [0, 1] and M.channel_counts(m) == (2, 1)
        assert not [p for p in M.validate(m) if p.severity == "error"], M.validate(m)
        m2 = dict(m, emg_channels="1", sample_rate="abc", ylim="auto")
        errs = {p.key for p in M.validate(m2) if p.severity == "error"}
        assert errs == {"emg_channels", "sample_rate", "ylim"}, errs
        m["project_path"] = None
        assert M.project_dir(m) == "."  # old code passed None to QFileDialog after 'reset'

        # ---- recordings: GUI layout [C, N, T] in chunks, last chunk tiny ----------------------
        rng = np.random.default_rng(0)
        rec = tmp / "rec1"
        rec.mkdir()
        c, t = 7, 1000
        sizes = [50, 50, 3]  # 3 windows < 7 channels: the old per-chunk rule transposed this one
        all_y = []
        for i, n in enumerate(sizes):
            x = rng.standard_normal((c, n, t)).astype(np.float32)
            x[1] *= 10  # channel 1 louder, to check channel order survives
            y = np.repeat(rng.integers(1, 5, n)[None, :, None], t, axis=2)  # [1, N, T] like the GUI saves
            np.save(rec / f"X_chunk{i}.npy", x)
            np.save(rec / f"y_chunk{i}.npy", y)
            all_y.append(y[0, :, 0])
        d = RecordingData.from_folder(rec)
        assert d.x.shape == (103, 7, 1000), d.x.shape
        assert np.array_equal(d.states, np.concatenate(all_y))
        assert d.x[:, 1].std() > 5 * d.x[:, 0].std()
        sig = d.signals(10, 3)
        assert sig.shape == (7, 3000) and np.array_equal(sig[:, 1000:2000], d.x[11])
        d.set_ylims("standard", 3)
        assert d.ylims[:3] == [(0.0, 0.2)] * 3 and d.ylims[3] == (0.5, 1.0)
        assert len(compute_ylims(d.x, "infer", 3)) == 7

        unl = tmp / "unlabeled"
        unl.mkdir()
        np.save(unl / "X_0.npy", rng.standard_normal((2, 20, 512)).astype(np.float32))
        u = RecordingData.from_folder(unl)
        assert not u.labeled and u.states.sum() == 0 and len(u) == 20

        # ---- annotations ------------------------------------------------------------------------
        tr = StateTrack.empty(20)
        assert tr.set_range(5, 2, 2) and tr.states[2:6].tolist() == [2] * 4 and tr.dirty
        assert not tr.set_range(2, 5, 2)  # no-op is not an edit
        tr.set_range(4, 8, 4)
        assert tr.undo() == (4, 8) and tr.states[2:6].tolist() == [2] * 4 and tr.states[6:9].sum() == 0
        assert tr.next_change(0) == 2 and tr.prev_change(10) == 6 and tr.next_unlabeled(3) == 6
        tr.confidence = np.linspace(1, 0, 20).astype(np.float32)
        assert tr.next_uncertain(0, 0.5) == 10
        fitted, msg = fit_states_to(np.ones(25, dtype=int), 20)
        assert len(fitted) == 20 and "dropped" in msg
        assert np.isnan(fit_array(np.ones(3, dtype=np.float32), 5, np.nan)[-1])
        sp = tmp / "s_states.pkl"
        with open(sp, "wb") as f:
            pickle.dump([1, 2, None, 4], f)
        with open(tmp / "s_states_confidence.pkl", "wb") as f:
            pickle.dump(np.array([0.9, 0.5, 0.4, 0.8]), f)
        assert load_states_file(sp).tolist() == [1, 2, 0, 4] and len(load_confidence_for(sp)) == 4

        # ---- agreement: REM must be compared (old num_classes count dropped code 4) ----------
        a = np.array([1, 1, 2, 2, 4, 4, 0, 2])
        b = np.array([1, 2, 2, 2, 4, 1, 1, 2, 2])
        r = compare_states(a, b)
        assert r["codes"] == [1, 2, 4] and r["windows_compared"] == 7 and "lengths differ" in r["note"]
        assert r["per_state"]["REM"]["A_windows"] == 2 and abs(r["accuracy"] - 5 / 7) < 1e-9
        assert "kappa" in format_agreement(r)
        r2 = compare_states(np.array([2, 3, 3]), np.array([2, 2, 3]), merge_is=True)
        assert r2["accuracy"] == 1.0

        # ---- autoscoring helpers ----------------------------------------------------------------
        assert source_rate_for(1000, 4.0, None)[0] == 250.0 and source_rate_for(512, 4.0, 128)[0] == 128
        try:
            source_rate_for(1000, 4.0, 1000)
            raise AssertionError("1-s windows must be refused")
        except ValueError:
            pass
        run = tmp / "run"
        (run / "heads" / "zeroshot").mkdir(parents=True)
        for f in ("enc_t.pt", "enc_z.pt", "heads/zeroshot/context_mlp.pt", "mlp_t.pt"):
            (run / f).write_bytes(b"x")
        (run / "manifest.json").write_text(json.dumps({"stages": {
            "embed@tuned": {"encoder_path": str(run / "enc_t.pt"), "encoder_stage": "adversarial"},
            "embed@zeroshot": {"encoder_path": str(run / "enc_z.pt"), "encoder_stage": "supcon"},
            "context_mlp@tuned": {"head": str(run / "mlp_t.pt")},
            "context_mlp@zeroshot": {"head": str(run / "heads/zeroshot/context_mlp.pt")},
            "supcon": {"final": "x"}}}))
        ms = discover_models(run)
        assert [x.mode for x in ms] == ["zeroshot", "tuned"]
        assert ms[0].encoder.endswith("enc_z.pt") and ms[1].encoder.endswith("enc_t.pt")
        parent = tmp / "many"
        for k in range(3):
            (parent / f"r{k}").mkdir(parents=True)
            np.save(parent / f"r{k}" / "X_0.npy", np.zeros((1, 4, 512), np.float32))
        assert len(expand_targets(parent)) == 3 and len(expand_targets(parent / "r1")) == 1
        assert output_path(tmp, Path("r1"), "context_mlp", "zeroshot", "v2").name == "r1_context_mlp_zeroshot_v2_states.pkl"

        # ---- preprocessing option checks ----------------------------------------------------------
        csv = tmp / "a.csv"
        csv.write_text("x")
        mm = dict(m, sample_rate="250", ecog_channels="0", emg_channels="1")
        o = PreprocessOptions(file=str(csv), save_windowed=True, win_len=1000)
        assert not check_options(o, mm)
        o = PreprocessOptions(file=str(csv), band_power=True, ratio_features=True, chunked=True, chunk_size=1500,
                              save_windowed=True, win_len=1000)
        levels = [lvl for lvl, _ in check_options(o, mm)]
        assert levels.count("error") == 1 and "confirm" in levels
        p = model_preset(mm)
        o = PreprocessOptions(file=str(csv), resample=True, new_sr=p["new_sr"], save_windowed=True, win_len=p["win_len"])
        assert not [x for x in check_options(o, mm) if x[0] != "info"]

        # ---- single-file data (storage.save_windowed non-chunked naming) ---------------------------
        from scorer.gui.core.autoscore import load_target, target_name
        from scorer.gui.core.recording import guess_label_file

        proc = tmp / "proj" / "processed"
        proc.mkdir(parents=True)
        xs = rng.standard_normal((3, 30, 1000)).astype(np.float32)
        np.save(proc / "20260101X1_mouseX_X.npy", xs)  # an 'X' inside the name fooled construct_paths
        np.save(proc / "20260101X1_mouseX_y.npy", np.full((1, 30, 1000), 2))
        assert guess_label_file(proc / "20260101X1_mouseX_X.npy").name == "20260101X1_mouseX_y.npy"
        one = RecordingData.from_file(proc / "20260101X1_mouseX_X.npy")
        assert one.labeled and one.states.tolist() == [2] * 30 and one.x.shape == (30, 3, 1000)
        assert load_target(proc / "20260101X1_mouseX_X.npy", 1).shape == (30, 1, 1000)
        assert target_name(proc / "20260101X1_mouseX_X.npy") == "20260101X1_mouseX"
        assert M.normalise_value("sample_rate", " 250.0 ") == "250" and M.normalise_value("ecog_channels", "0, 1") == "0,1"
        assert any(p.key == "sample_rate" for p in M.validate(dict(m, sample_rate="250.5")))

        # ---- training core -------------------------------------------------------------------------
        from scorer.gui.core import training as T

        assert {"pipeline_128hz", "pipeline_250hz"} <= set(T.templates())
        chk = T.check_config(T.templates()["pipeline_128hz"])
        assert chk.cfg is not None and chk.default_stages[0] == "inspect"  # dataset folders are on your machine
        cmd = T.build_command("c.toml", ["embed", "evaluate"], T.build_overrides("r 1", "zeroshot", "cpu", "x.y=1"))
        assert cmd[1:4] == ["-u", "-m", "scorer.pipeline.run"] and 'run.name="r 1"' in cmd and "x.y=1" in cmd
        pr = T.Progress(stages=["prepare", "simclr"])
        for line in ("==================== stage simclr ====================", "simclr epoch 3/50: loss 2.0",
                     "stage simclr done in 9s", "all stages finished: ['simclr']"):
            pr.feed(line)
        assert pr.done == ["simclr"] and pr.finished and pr.step == (0, 0)

        # ---- plots ---------------------------------------------------------------------------------
        assert parse_rec_start("2025-11-27 19:16:00.301587302\n")[0] == 19 * 3600 + 16 * 60 + 0.301587
        assert parse_rec_start(None) == (0.0, False)
        fmt = TimeOfDayFormatter(23 * 3600 + 59 * 60, 30)
        assert fmt(40) == "00:00:10"
        view = SignalView(7, ["ecog"] * 2, ["me", "model"], fmt, spectrogram=True)
        view.update(d.signals(0, 100), d.ylims, 250.0, {"me": d.states[:100], "model": d.states[:100]}, 4.0,
                    np.linspace(0, 1, 100), spec_channel=0)
        x, y = view.state_lines["me"].get_data()
        assert len(x) == 101 and x[-1] == 400.0  # one point per window (old: one per sample = 100,000)
        view.update(d.signals(5, 1), d.ylims, 250.0, {"me": d.states[5:6]}, 4.0)
        assert view.label_text.get_text().startswith("State:")
        fig = hypnogram(d.states, np.arange(len(d.states)) * 4.0, confidence=np.ones(len(d.states)))
        fig.savefig(tmp / "h.png")
        confusion_figure(r).savefig(tmp / "c.png")
        print("[gui core] all checks passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
