#custom funcs for reading from https://github.com/jenniferColonell/SpikeGLX_Datafile_Tools/
from pathlib import Path
import numpy as np
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
    if meta['typeThis'] == 'imec':
        AP, LF, SY = ChannelCountsIM(meta)
        if SY == 0:
            print("No imec sync channel saved.")
            digArray = np.zeros((0), 'uint8')
            return(digArray)
        else:
            digCh = AP + LF + dwReq
    elif meta['typeThis'] == 'nidq':
        MN, MA, XA, DW = ChannelCountsNI(meta)
        if dwReq > DW-1:
            print("Maximum digital word in file = %d" % (DW-1))
            digArray = np.zeros((0), 'uint8')
            return(digArray)
        else:
            digCh = MN + MA + XA + dwReq
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