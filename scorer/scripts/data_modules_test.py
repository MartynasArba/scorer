"""
Checks for the GUI's data modules that run without torch / a GPU:
storage time and path helpers, report, OneBox conversion and quality report, motion-sensor split.

    python -m scorer.scripts.data_modules_test
"""
from __future__ import annotations

import io
import shutil
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    """Run all checks; raises AssertionError on the first failure, prints a line when all pass."""
    import matplotlib

    matplotlib.use("Agg")
    from scipy.signal import resample_poly

    from scorer.data import onebox_utils as O
    from scorer.data import storage as S
    from scorer.data.motionsensor_utils import parse_sensors
    from scorer.data.report import generate_sleep_report, label_microawakenings

    tmp = Path(tempfile.mkdtemp(prefix="data_mods_"))
    quiet = io.StringIO()
    try:
        # ---- storage -------------------------------------------------------------------------
        assert S.resolve_time_window("2025-11-27 19:16:00", ("19:30:00", "21:00:00"), 1000)[:2] == (840_000, 6_240_000)
        start, end, new = S.resolve_time_window("2025-11-27 19:16:00", ("01:00:00", "05:00:00"), 10)
        assert (start, end) == (206_400, 350_400) and new.startswith("2025-11-28 01:00"), (start, end, new)
        assert S.resolve_time_window("2025-11-27 19:16:00", ("19:00:00", "07:00:00"), 1) == (0, 42_240, None)
        f = tmp / "r.csv"
        pd.DataFrame({"a": np.arange(20), "b": np.arange(20) * 10}).to_csv(f, index=False)
        w = S._read_csv_window(f, 5, 8)
        assert list(w.columns) == ["a", "b"] and w["a"].tolist() == [5, 6, 7]
        ta = S.get_timearray_for_states(np.zeros(7), 512, {"sample_rate": 250, "rec_start": "2025-11-27 19:16:00.3\n"})
        assert len(ta) == 7 and (ta[1] - ta[0]).total_seconds() == 2.048
        (tmp / "proj" / "processed" / "windowed_a").mkdir(parents=True)
        _, p1 = S.construct_paths(tmp / "proj" / "processed" / "windowed_a", add="autosave")
        assert p1.parent == tmp / "proj" / "scores" and "None" not in p1.name
        try:
            S.construct_paths(tmp / "nope")
            raise AssertionError("missing path must raise")
        except FileNotFoundError:
            pass
        S.save_metadata(tmp / "m.json", {"x": np.float32(1.5), "p": tmp, "none": None})
        assert S.load_metadata(tmp / "m.json")["none"] is None
        assert S._sample_rate({"sample_rate": "250.0"}, 1) == 250

        # ---- report ------------------------------------------------------------------------------
        st = np.array([1] * 900 + [2] * 1800 + [4] * 300 + [2] * 600 + [6] * 10)
        ta = S.get_timearray_for_states(st, 1000, {"sample_rate": 250, "rec_start": "2025-11-27 22:30:00"})
        with redirect_stdout(quiet):
            r = generate_sleep_report(list(st), ta, get_by_hour=True, win_len=1000, metadata={"sample_rate": "250"})
        assert list(r) == ["overall", "22-23", "23-0", "0-1", "1-2"], list(r)  # chronological, last partial hour out
        assert r["overall"]["6_duration"] == 40.0
        assert label_microawakenings(np.array([2, 2, 1, 1, 2, 2, 1, 2])).tolist() == [2, 2, 5, 5, 2, 2, 5, 2]

        # ---- OneBox: chunked resampling == whole-signal resampling --------------------------------
        rng = np.random.default_rng(0)
        for sr_in, sr_out, n in ((1e6 / 33, 1000, 400_003), (1000, 250, 123_457)):
            x = rng.standard_normal((3, n)).astype(np.float32)
            up, down = O.compute_rational_ratio(sr_in, sr_out)
            ref = resample_poly(x.astype(np.float64), up, down, axis=1)
            y = O.downsample_memmap_multichannel(x, sr_in, sr_out, chunk_size=50_000, verbose=False)
            assert y.shape == ref.shape and np.abs(y - ref).max() < 1e-5, (y.shape, ref.shape)
        nchan, nsamp = 16, 300_000
        data = (rng.standard_normal((nchan, nsamp)) * 1000).astype(np.int16)
        bin_path = tmp / "obx" / "rec_g0" / "rec_g0_t0.obx0.obx.bin"
        bin_path.parent.mkdir(parents=True)
        data.T.copy().tofile(bin_path)  # file order: sample-major (memmap order='F')
        bin_path.with_suffix(".meta").write_text("\n".join([
            "typeThis=obx", f"obSampRate={1e6 / 33}", "obMaxInt=32768", "obAiRangeMax=5", f"nSavedChans={nchan}",
            f"fileSizeBytes={nchan * nsamp * 2}", "firstSample=0", "fileCreateTime=2025-11-27T19:16:00", "", "x"]))
        with redirect_stdout(quiet):
            summary = O.convert_multiple_recs(str(tmp / "obx"), {"project_path": str(tmp)}, save_folder=str(tmp / "csv"),
                                              sr_new=1000)
        assert len(summary["converted"]) == 1 and not summary["failed"], summary
        df = pd.read_csv(tmp / "csv" / "rec_g0_t0.obx0.obx_box1.csv")
        exact = resample_poly(data[0:3].astype(np.float64), 33, 1000, axis=1) * (5 / 32768)
        assert np.abs(df[["f_ecog", "p_ecog", "emg"]].values.T - exact).max() < 1e-6
        with redirect_stdout(quiet):
            again = O.convert_multiple_recs(str(tmp / "obx"), {}, save_folder=str(tmp / "csv"), sr_new=1000)
            rep = O.get_folder_quality_report(str(tmp / "csv"), savepath=str(tmp), save_fig=True, sr=1000)
        assert len(again["skipped"]) == 1 and pd.read_csv(rep).shape[0] == 4
        assert len(list((tmp / "quality_plots").glob("*.png"))) == 4

        # ---- motion sensors: names follow the data ---------------------------------------------------
        ms = tmp / "motion"
        ms.mkdir()
        pd.DataFrame([[2, 1, 2], [3, 5, 6], [2, 7, 8]]).to_csv(ms / "MOTIONrecSLEEPECOG_A.csv", header=False, index=False)
        pd.DataFrame([[3, 9, 9], [5, 1, 1], [9, 0, 0]]).to_csv(ms / "MOTIONrecSLEEPECOG_B.csv", header=False, index=False)
        with redirect_stdout(quiet):
            names = sorted(p.name for p in parse_sensors(str(ms)))
        assert names == ["cage_1_MOTIONrecSLEEPECOG_A.csv", "cage_2_MOTIONrecSLEEPECOG_A.csv",
                         "cage_2_MOTIONrecSLEEPECOG_B.csv", "cage_4_MOTIONrecSLEEPECOG_B.csv"], names
        print("[data modules] all checks passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
