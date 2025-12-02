import pandas as pd
from pathlib import Path
import glob
import os

def parse_sensors(path = 'G:/SLEEP-ECOG/'):
    cages = {}
    if not os.path.isdir(path):
        folder = Path(path).parent
    else:
        folder = path
        
    cages, suffix = convert_all_files(folder = folder, cages = cages)
    save_sensors(folder, cages = cages, suffix = suffix)
    
def convert_all_files(folder, cages = {}, return_suffix = True):
    suffix = []
    files = glob.glob(folder + '/MOTIONrecSLEEPECOG*.csv')
    for file in files:
        cages = load_sensor_file(file, cages = cages)
        if return_suffix:
            suffix.append(Path(file).stem)
    if not suffix:  #clean up
        suffix = None            
    return cages, suffix
    
def load_sensor_file(path, cages = {}):
    pin_to_cage = {i: i - 1 for i in range(2, 6)}
    try:
        data = pd.read_csv(path, header = None)
    except Exception as e:
        print(f'failed to load file {path}: {e}')
        return cages
    
    for pin in data[0].unique():
        cage = pin_to_cage[pin]
        df = data.loc[data[0] == pin].drop(0, axis=1)

        if cage not in cages:
            cages[cage] = [df]
        else:
            cages[cage].append(df)
    return cages

def save_sensors(folder, cages, suffix = None):
    folder = Path(folder)
    for id, data_list in cages.items():
        for i, data in enumerate(data_list):
            if isinstance(suffix, list):
                savepath = folder / f'cage_{id}_{suffix[i]}.csv'
            elif suffix is not None:
                savepath = folder / f'cage_{id}_{suffix}.csv'
            else:
                savepath = folder / f'cage_{id}.csv'
            data.to_csv(savepath)

if __name__ == "__main__":
    parse_sensors()