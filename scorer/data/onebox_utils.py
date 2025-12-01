#custom funcs for reading onebox files 
# a lot comes from https://github.com/jenniferColonell/SpikeGLX_Datafile_Tools/
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from fractions import Fraction
import glob
import os

# Parse ini file returning a dictionary whose keys are the metadata
# left-hand-side-tags, and values are string versions of the right-hand-side
# metadata values. We remove any leading '~' characters in the tags to match
# the MATLAB version of readMeta.
#
# The string values are converted to numbers using the "int" and "float"
# functions. Note that python 3 has no size limit for integers.
#
def readMeta(binFullPath):
    metaName = binFullPath.stem + ".meta"
    metaPath = Path(binFullPath.parent / metaName)
    metaDict = {}
    if metaPath.exists():
        # print("meta file present")
        with metaPath.open() as f:
            mdatList = f.read().splitlines()
            # convert the list entries into key value pairs
            for m in mdatList:
                csList = m.split(sep='=')
                if csList[0][0] == '~':
                    currKey = csList[0][1:len(csList[0])]
                else:
                    currKey = csList[0]
                metaDict.update({currKey: csList[1]})
    else:
        print("no meta file")
    return(metaDict)


# Return sample rate as python float.
# On most systems, this will be implemented as C++ double.
# Use python command sys.float_info to get properties of float on your system.
#
def SampRate(meta):
    if meta['typeThis'] == 'imec':
        srate = float(meta['imSampRate'])
    elif meta['typeThis'] == 'nidq':
        srate = float(meta['niSampRate'])
    elif meta['typeThis'] == 'obx':
        srate = float(meta['obSampRate'])
    else:
        print('Error: unknown stream type')
        srate = 1
        
    return(srate)


# Return a multiplicative factor for converting 16-bit file data
# to voltage. This does not take gain into account. The full
# conversion with gain is:
#         dataVolts = dataInt * fI2V / gain
# Note that each channel may have its own gain.
#
def Int2Volts(meta):
    if meta['typeThis'] == 'imec':
        if 'imMaxInt' in meta:
            maxInt = int(meta['imMaxInt'])
        else:
            maxInt = 512
        fI2V = float(meta['imAiRangeMax'])/maxInt
    elif meta['typeThis'] == 'nidq':
        maxInt = int(meta['niMaxInt'])
        fI2V = float(meta['niAiRangeMax'])/maxInt
    elif meta['typeThis'] == 'obx':
        maxInt = int(meta['obMaxInt'])
        fI2V = float(meta['obAiRangeMax'])/maxInt
    else:
        print('Error: unknown stream type')
        fI2V = 1
        
    return(fI2V)


# Return array of original channel IDs. As an example, suppose we want the
# imec gain for the ith channel stored in the binary data. A gain array
# can be obtained using ChanGainsIM(), but we need an original channel
# index to do the lookup. Because you can selectively save channels, the
# ith channel in the file isn't necessarily the ith acquired channel.
# Use this function to convert from ith stored to original index.
# Note that the SpikeGLX channels are 0 based.
#
def OriginalChans(meta):
    if meta['snsSaveChanSubset'] == 'all':
        # output = int32, 0 to nSavedChans - 1
        chans = np.arange(0, int(meta['nSavedChans']))
    else:
        # parse the snsSaveChanSubset string
        # split at commas
        chStrList = meta['snsSaveChanSubset'].split(sep=',')
        chans = np.arange(0, 0)  # creates an empty array of int32
        for sL in chStrList:
            currList = sL.split(sep=':')
            if len(currList) > 1:
                # each set of contiguous channels specified by
                # chan1:chan2 inclusive
                newChans = np.arange(int(currList[0]), int(currList[1])+1)
            else:
                newChans = np.arange(int(currList[0]), int(currList[0])+1)
            chans = np.append(chans, newChans)
    return(chans)

# Return counts of each obx channel type that composes the timepoints
# stored in the binary files.
#
def ChannelCountsOBX(meta):
    chanCountList = meta['snsXaDwSy'].split(sep=',')
    XA = int(chanCountList[0])
    DW = int(chanCountList[1])
    SY = int(chanCountList[2])
    return(XA, DW, SY)


def GainCorrectOBX(dataArray, chanList, meta):

    fI2V = Int2Volts(meta)

    # make array of floats to return. dataArray contains only the channels
    # in chanList, so output matches that shape
    convArray = np.zeros(dataArray.shape, dtype=float)
    for i in range(0, len(chanList)):
        # dataArray contains only the channels in chanList
        convArray[i, :] = dataArray[i, :] * fI2V
    return(convArray)

# Return memmap for the raw data
# Fortran ordering is used to match the MATLAB version
# of these tools.
#
def makeMemMapRaw(binFullPath, meta):
    nChan = int(meta['nSavedChans'])
    nFileSamp = int(int(meta['fileSizeBytes'])/(2*nChan))
    print("nChan: %d, nFileSamp: %d" % (nChan, nFileSamp))
    rawData = np.memmap(binFullPath, dtype='int16', mode='r',
                        shape=(nChan, nFileSamp), offset=0, order='F')
    return(rawData)


# Return an array [lines X timepoints] of uint8 values for a
# specified set of digital lines.
#
# - dwReq is the zero-based index into the saved file of the
#    16-bit word that contains the digital lines of interest.
# - dLineList is a zero-based list of one or more lines/bits
#    to scan from word dwReq.
#
def ExtractDigital(rawData, firstSamp, lastSamp, dwReq, dLineList, meta):
    # Get channel index of requested digital word dwReq
    if not meta['typeThis'] == 'obx':
        print('non-obx files not currenly supported in onebox_utils.py')
    # if meta['typeThis'] == 'imec':
    #     AP, LF, SY = ChannelCountsIM(meta)
    #     if SY == 0:
    #         print("No imec sync channel saved.")
    #         digArray = np.zeros((0), 'uint8')
    #         return(digArray)
    #     else:
    #         digCh = AP + LF + dwReq
    # elif meta['typeThis'] == 'nidq':
    #     MN, MA, XA, DW = ChannelCountsNI(meta)
    #     if dwReq > DW-1:
    #         print("Maximum digital word in file = %d" % (DW-1))
    #         digArray = np.zeros((0), 'uint8')
    #         return(digArray)
    #     else:
    #         digCh = MN + MA + XA + dwReq
    elif meta['typeThis'] == 'obx':
        XA, DW, SY = ChannelCountsOBX(meta)
        if dwReq > DW-1:
            print("Maximum digital word in file = %d" % (DW-1))
            digArray = np.zeros((0), 'uint8')
            return(digArray)
        else:
            digCh = XA + dwReq
    else:
        print('unknown data stream')

    selectData = np.ascontiguousarray(rawData[digCh, firstSamp:lastSamp+1], 'int16')
    nSamp = lastSamp-firstSamp + 1

    # unpack bits of selectData; unpack bits works with uint8
    # original data is int16
    bitWiseData = np.unpackbits(selectData.view(dtype='uint8'))
    # output is 1-D array, nSamp*16. Reshape and transpose
    bitWiseData = np.transpose(np.reshape(bitWiseData, (nSamp, 16)))

    nLine = len(dLineList)
    digArray = np.zeros((nLine, nSamp), 'uint8')
    for i in range(0, nLine):
        byteN, bitN = np.divmod(dLineList[i], 8)
        targI = byteN*8 + (7 - bitN)
        digArray[i, :] = bitWiseData[targI, :]
    return(digArray)

def compute_rational_ratio(sr_in, sr_out, max_den=10_000):
    """
    Compute rational approximation U/D ≈ sr_out / sr_in.
    max_den controls accuracy & speed.
    """
    if not isinstance(sr_out, int) or not isinstance(sr_in, int):
        try:
            sr_in, sr_out = int(sr_in), int(sr_out)
        except:
            print('sr_in and sr_out have to be int!')
    
    frac = Fraction(sr_out, sr_in).limit_denominator(max_den)
    return frac.numerator, frac.denominator   # up, down

def downsample_memmap_multichannel(
        data,
        sr_in=30303,
        sr_out=1000,
        chunk_size=5_000_000,
        max_den=10_000,
        verbose=True
    ):
    """
    Downsample multi-channel onebox recordings (np.arrays of channel x time) 
    using polyphase FIR filtering (scipy.signal.resample_poly)
    """

    C, T = data.shape

    # Compute rational approximation up/down
    up, down = compute_rational_ratio(sr_in, sr_out, max_den=max_den)
    if verbose:
        print(f"Resampling ratio: sr_out/sr_in ≈ {sr_out/sr_in:.8f}")
        print(f"Using rational up={up}, down={down}  (error={abs((up/down) - (sr_out/sr_in)):.3e})")

    # Output length
    T_out = int(np.floor(T * (sr_out / sr_in)))
    if verbose:
        print(f"Output samples: {T_out:,}")
        
    y = np.empty((C, T_out))

    # Overlap region for FIR filter edge safety
    overlap = 300   # safe for 100–1000 tap FIR

    out_pos = 0

    for start in range(0, T, chunk_size):
        stop = min(T, start + chunk_size)

        # Add overlap on both sides for edge-safe filtering
        s = max(0, start - overlap)
        e = min(T, stop + overlap)

        if verbose:
            print(f"Processing time samples {s:,} → {e:,}")

        # Extract chunk
        chunk = data[:, s:e]   # shape: [C, time]

        # Polyphase resampling
        chunk_ds = resample_poly(chunk, up=up, down=down, axis=1)

        # Compute corresponding overlap in output domain
        overlap_out = int(np.ceil(overlap * (sr_out / sr_in)))

        # Remove overlap edges
        if start == 0:
            core = chunk_ds[:, :-(overlap_out)] if stop < T else chunk_ds
        elif stop == T:
            core = chunk_ds[:, overlap_out:]
        else:
            core = chunk_ds[:, overlap_out:-overlap_out]

        L = core.shape[1]

        # Write
        y[:, out_pos:out_pos + L] = core
        out_pos += L

    if verbose:
        print("done")

    return y

def run_conversion(bin_path: str, project_meta: dict,  
                   sr_new: int = 1000, 
                   chanlist = [x for x in range(12)],
                   channel_to_box = {0:1, 1:4, 2:2, 3:3}):
    """
    converts obx bin file to 4 csv files
    some params can be changed if the setup is different
    uses only chanlist first channels
    """
    bin_path = Path(bin_path)
    print('path read')
    #read data
    meta = readMeta(Path(bin_path))
    sr_obx = SampRate(meta)
    raw_data = makeMemMapRaw(Path(bin_path), meta)
    print('data loaded, downsampling...')
    downsampled = downsample_memmap_multichannel(data = raw_data, sr_in = sr_obx, sr_out = sr_new)
    #gain correct
    conv_obx = GainCorrectOBX(downsampled, chanlist, meta)
    #get time axis
    time = np.arange(int(meta['firstSample']), 
                     conv_obx.shape[1] + int(meta['firstSample']), 1) / sr_new
    timestamps = pd.to_datetime(meta['fileCreateTime']) + pd.to_timedelta(time, unit = 's') #probably no getting around this, np.vectorize with datetime is slower, np.datetime is int only
    #print info to troubleshoot
    print(f'recording duration: {time[-1] - time[0]} seconds')
    print(f'loaded {conv_obx.shape} data points, created time array: {timestamps.shape}')
    
    save_folder = Path(project_meta.get('project_path', '.')) / 'raw'
    for i in range(4):
        print(f'saving {i+1} / {4} csv')
        filename = bin_path.stem + f'_box{channel_to_box.get(i,'unknown')}.csv'
        save_path = save_folder / filename
        sel_data = conv_obx[i*3:i*3+3, :]
        to_save = pd.DataFrame(sel_data.T, columns = ['f_ecog', 'p_ecog', 'emg'])
        to_save['time'] = timestamps
        to_save.to_csv(save_path, index = False)
    print('saved!')
    return

def file_converted(file, save_folder = "C:/Users/marty/Projects/scorer/proj_data/raw", channel_to_box = {0:1, 1:4, 2:2, 3:3}):
    """
    checks if obx file is already converted
    converted if there are corresponding .csv files
    """
    bin_path = Path(file)
    
    for i in range(4):
        filename = bin_path.stem + f'_box{channel_to_box.get(i,'unknown')}.csv'
        save_path = save_folder / filename
        if os.path.exists(save_path):
            return True
    return False   
    
def convert_multiple_recs(folder, project_meta, overwrite = False):
    """
    runs conversion obx.bin -> 4 recs .csv for all files in folder
    """
    print(f'starting all file conversion in {folder}')
    files = glob.glob(folder + '/*/*.obx0.obx.bin')
    for file in files:
        if (not file_converted(file, save_folder= Path(project_meta.get('project_path', '.')) / 'raw')) or overwrite:
            try:
                print(f'converting {file}')
                run_conversion(file, project_meta, sr_new = 1000, 
                            chanlist = [x for x in range(12)],
                            channel_to_box = {0:1, 1:4, 2:2, 3:3})
            except Exception as e:
                print(f'{file} NOT CONVERTED: {e}')
        else:
            print(f'file already converted: {file}')
    print(f'all files in {folder} converted!!!')
    return

def get_folder_quality_report(folder_path, savepath = None):
    """
    generates a quality report for all .csv files in path
    """
    if savepath is None:
        savepath = folder_path
    res_dict = {}
    files = glob.glob(folder_path + '/*box*.csv')
    for file in files:
        metrics = generate_obx_quality_report(file, sample_size = 20, report_interval = 10, sr = 1000)
        res_dict[file] = metrics
        print(f'report generated for {file}')    
    pd.DataFrame.from_dict(res_dict).to_csv(savepath + '/quality_report.csv')

def generate_obx_quality_report(path, sample_size = 20, report_interval = 10, sr = 1000):
    """
    generates a quality report for a raw .csv recording file obtained from obx in earlier steps+
    file has to contain f_ecog, p_ecog, emg channels
    """
    channels = ['f_ecog', 'p_ecog', 'emg']
    metrics = {f'std_{channel}':[] for channel in channels}
    metrics['time'] = []
    data = pd.read_csv(path, chunksize = sample_size * sr)      #20 sec per hour
    for i, chunk in enumerate(data):
        if i % int(report_interval * sample_size) == 0:
            print(f'{i} chunk read')
            metrics['time'].append(chunk['time'].iloc[0]) #should be a single value
            for channel in channels:
                metrics[f'std_{channel}'].append(chunk.loc[:, channel].std())            
    return metrics

if __name__ == "__main__":
    project_meta = {'project_path' : 'C:/Users/marty/Projects/scorer/proj_data', 'sample_rate' : 1000}
    convert_multiple_recs(folder = 'G:/SLEEP-ECOG', project_meta = project_meta, overwrite = False)
    get_folder_quality_report("C:/Users/marty/Projects/scorer/proj_data/raw")

