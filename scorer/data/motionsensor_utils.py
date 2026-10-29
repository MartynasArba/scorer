"""
Split motion-sensor recordings (one csv per recording, rows tagged with an Arduino pin) into one
csv per cage.

Fixed in this revision:
  * output names were taken from suffix[i], where i counted only the files in which THAT cage
    appeared; when a cage was missing from one file, its later data was saved under the wrong
    recording's name (and the last file's name was never used);
  * a pin outside 2-5 raised KeyError and aborted the whole folder; it is now skipped with a warning;
  * a file path (instead of a folder) crashed on `Path + str`;
  * mutable default arguments ({} shared between calls) replaced;
  * parse_sensors returns what it wrote.
"""
import glob
import os
from pathlib import Path

import pandas as pd

PIN_TO_CAGE = {i: i - 1 for i in range(2, 6)}


def parse_sensors(path: str = 'G:/SLEEP-ECOG/') -> list[Path]:
    """splits every MOTIONrecSLEEPECOG*.csv in the folder into cage_<id>_<recording>.csv files"""
    folder = str(path) if os.path.isdir(path) else str(Path(path).parent)
    cages, suffix = convert_all_files(folder=folder)
    return save_sensors(folder, cages=cages, suffix=suffix)


def convert_all_files(folder: str, cages: dict | None = None, return_suffix: bool = True) -> tuple[dict, list | None]:
    """cages: {cage_id: [(recording_name, dataframe), ...]}"""
    cages = {} if cages is None else cages
    files = sorted(glob.glob(str(folder) + '/MOTIONrecSLEEPECOG*.csv'))
    print(f'{len(files)} files found')
    for file in files:
        cages = load_sensor_file(file, cages=cages)
    return cages, ([Path(f).stem for f in files] or None) if return_suffix else None


def load_sensor_file(path: str, cages: dict | None = None) -> dict:
    """adds (recording name, data) of every cage in one sensor file to `cages`"""
    cages = {} if cages is None else cages
    try:
        data = pd.read_csv(path, header=None)
    except Exception as e:
        print(f'failed to load file {path}: {e}')
        return cages
    name = Path(path).stem
    for pin in data[0].unique():
        cage = PIN_TO_CAGE.get(pin)
        if cage is None:
            print(f'{path}: unknown pin {pin!r} skipped')
            continue
        cages.setdefault(cage, []).append((name, data.loc[data[0] == pin].drop(0, axis=1)))
    return cages


def save_sensors(folder: str, cages: dict, suffix=None) -> list[Path]:
    """writes cage_<id>_<recording>.csv; `suffix` is kept for compatibility (names now travel with the data)"""
    folder = Path(folder)
    written = []
    for cage_id, entries in cages.items():
        for entry in entries:
            name, data = entry if isinstance(entry, tuple) else (None, entry)
            savepath = folder / (f'cage_{cage_id}_{name}.csv' if name else f'cage_{cage_id}.csv')
            data.to_csv(savepath)
            written.append(savepath)
    return written


if __name__ == "__main__":
    parse_sensors(path=r'G:\sleep-ecog-DOWNSAMPLED\motion')
