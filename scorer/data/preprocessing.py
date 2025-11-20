import math
import torch
import torch.nn.functional as F

#for filters widmann  et al. 2015 was consulted, settled on FIR filter, Hamming window, FFT implementation for efficiency
def design_bandpass_hamming(fs, lowcut, highcut, 
                            transition_width = 1.0, 
                            numtaps = None, 
                            dtype = torch.float32, 
                            device = 'cuda'):
    """
    Design a linear-phase bandpass FIR using windowed sinc (Hamming).
    Returns a 1D tensor of taps (odd length).
    """
    #checks 
    nyq = fs / 2.0
    if lowcut <= 0:
        raise ValueError("lowcut must be > 0 for high-pass part of bandpass")
    if highcut >= nyq:
        raise ValueError("highcut must be < Nyquist (fs/2)")
    #numtaps = filter length/num of coefficients
    # If numtaps not given, estimate from transition_width using Hamming approx
    # Hamming window approx transition width Δf ≈ 3.3 / N (normalized to Nyquist units),
    # so N ≈ 3.3 / (Δf/nyq) = 3.3 * nyq / Δf
    if numtaps is None: 
        numtaps = int(math.ceil(3.3 * nyq / transition_width))
    # ensure odd
    if numtaps % 2 == 0:
        numtaps += 1
    M = numtaps
    m = torch.arange(M, dtype=dtype, device=device) - (M - 1) / 2.0  # symmetric time index centered at zero

    # normalized cutoff frequencies (radians/sample)
    wl = 2.0 * math.pi * lowcut / fs
    wh = 2.0 * math.pi * highcut / fs

    # ideal bandpass impulse response = (sin(wh*n) - sin(wl*n)) / (pi*n)
    taps = torch.empty(M, dtype=dtype, device=device)
    # avoid division by zero
    m_nonzero = m.clone()
    m_nonzero[torch.abs(m_nonzero) < 1e-12] = 1.0

    h_wh = torch.sin(wh * m_nonzero) / (math.pi * m_nonzero)
    h_wl = torch.sin(wl * m_nonzero) / (math.pi * m_nonzero)
    taps = h_wh - h_wl

    # correct center sample (m==0) to (wh - wl) / pi
    center_idx = (M - 1) // 2
    taps[center_idx] = (wh - wl) / math.pi

    # apply Hamming window
    n = torch.arange(M, device=device, dtype=dtype)
    hamming = 0.54 - 0.46 * torch.cos(2.0 * math.pi * n / (M - 1))
    taps = taps * hamming

    # normalize by the maximal magnitude of the FFT of taps for better passband scaling).
    # Compute small FFT on CPU to find max magnitude (cheap because numtaps moderate)
    with torch.no_grad():
        # use zero padding to do fine FFT resolution
        fftlen = 4096
        H = torch.fft.rfft(taps, n=fftlen)
        mag = torch.abs(H)
        max_mag = float(mag.max())
        if max_mag > 0:
            taps = taps / max_mag
            
    return taps  # on device

def design_notch_hamming(fs, notch_center, notch_width=2.0, transition_width=1.0,
                         numtaps=None, dtype=torch.float32, device='cuda'):
    """
    designs a linear-phase FIR notch (band-stop) filter using Hamming window 
    """

    nyq = fs / 2
    f1 = notch_center - notch_width/2
    f2 = notch_center + notch_width/2

    if f1 <= 0 or f2 >= nyq:
        raise ValueError("Notch band must be inside Nyquist range")

    # Estimate numtaps if not given
    if numtaps is None:
        # Hamming window rule: N ≈ 3.3 * Nyquist / transition_width
        numtaps = int(math.ceil(3.3 * nyq / transition_width))
    if numtaps % 2 == 0:
        numtaps += 1

    M = numtaps
    m = torch.arange(M, dtype=dtype, device=device) - (M - 1) / 2

    # Convert to radians/sample
    w1 = 2 * math.pi * f1 / fs
    w2 = 2 * math.pi * f2 / fs

    # Ideal bandstop = LP(w1) + HP(w2)
    eps = 1e-20
    m_nz = m.clone()
    m_nz[torch.abs(m_nz) < 1e-12] = 1.0

    h_lp_w1 = torch.sin(w1 * m_nz) / (math.pi * m_nz)
    h_lp_w2 = torch.sin(w2 * m_nz) / (math.pi * m_nz)
    h_bs = h_lp_w1 + (torch.zeros_like(h_lp_w1) - h_lp_w2)

    # Correct center sample
    c = (M - 1)//2
    h_bs[c] = (w1 - 0) / math.pi + (math.pi - w2)/math.pi

    # Hamming window
    n = torch.arange(M, device=device, dtype=dtype)
    hamming = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (M - 1))
    h_bs = h_bs * hamming

    # Normalize passband gain
    with torch.no_grad():
        fftlen = 4096
        H = torch.fft.rfft(h_bs, n=fftlen)
        max_mag = float(torch.abs(H).max())
        h_bs = h_bs / max_mag

    return h_bs

def fft_convolve_1d(x, h):
    """
    Linear convolution via FFT on GPU.
    x: (..., time) - 1 channel data
    h: (K,) - filter kernel
    Returns same dtype/device.
    """
    T = x.shape[-1]
    K = h.shape[-1]
    L = T + K - 1   #length of convolution L = time + kernel_length -1 
    n_fft = 1 << (L - 1).bit_length()   #padding to at least length, rounded up to 2^x for computation speed
    X = torch.fft.rfft(x, n=n_fft)  #freq-magnitude domain of signal
    H = torch.fft.rfft(h, n=n_fft)  #freq-magnitude domain of filter
    Y = X * H   #freq domain convolution
    y = torch.fft.irfft(Y, n=n_fft)[..., :L]    #inverse fft, but slice to L to remove padding
    return y

def fft_filtfilt(x, taps, padlen_factor=3):
    """
    Zero-phase (forward-reverse) filtering using FFT-based convolution (uses fft_convolve_1d)
    x: (batch, channels, time) or (channels, time) or (time,)
    taps: (numtaps,) symmetric FIR filter
    padlen same as SciPy default
    """
    orig_shape = x.shape
    single_channel = False
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0)
        single_channel = True
    elif x.ndim == 2:
        x = x.unsqueeze(0)

    device = x.device
    dtype = x.dtype
    taps = taps.to(device=device, dtype=dtype)
    k = taps.shape[0]
    padlen = padlen_factor * (k - 1)

    B, C, T = x.shape   #batch/channels/time
    if T <= padlen:
        raise ValueError("Input too short for pad length.")
    
    left = x[:, :, 1:padlen+1].flip(-1)  # reflective pad on both sides (mirrored signals) to prevent transient effects
    right = x[:, :, -padlen-1:-1].flip(-1)
    x_ext = torch.cat([left, x, right], dim=-1)  # (B,C,T+2*padlen)

    def conv_func(sig):         #local helper to apply convolution to every channel separately
        return fft_convolve_1d(sig, taps)

    y = torch.stack([conv_func(x_ext[:, i, :]) for i in range(C)], dim=1) # stack convolutions of each channel

    # reverse to get zero phase
    y_rev = torch.flip(y, dims=[-1])
    y2 = torch.stack([conv_func(y_rev[:, i, :]) for i in range(C)], dim=1)
    y_out_ext = torch.flip(y2, dims=[-1])

    #removing padding
    start_idx = padlen - (k - 1)
    end_idx = start_idx + T
    start_idx = max(start_idx, 0)   
    end_idx = min(end_idx, y_out_ext.shape[-1])
    y_final = y_out_ext[..., start_idx:end_idx] #match length to input

    # adjust length if off by one
    if y_final.shape[-1] != T:
        diff = T - y_final.shape[-1]
        if diff > 0:
            pad_val = y_final[..., -1:].expand(B, C, diff)
            y_final = torch.cat([y_final, pad_val], dim=-1)
        else:
            y_final = y_final[..., :T]
    #restore original shape
    if single_channel:
        return y_final.squeeze(0).squeeze(0)
    elif len(orig_shape) == 2:
        return y_final.squeeze(0)
    else:
        return y_final

def bandpass_filter(signal: torch.Tensor,
                        sr: int = 250,
                        freqs: tuple = (0.1, 49.0),
                        device: str = 'cuda') -> torch.Tensor:
    """
    integrates previous funcs
    FIR bandpass filter via FFT convolution on GPU. signal: [T], [C, T], [batch, channel, time]
    hopefully fast!
    """   
    taps = design_bandpass_hamming(sr, freqs[0], freqs[1], transition_width = 1.0, device = device)
    y = fft_filtfilt(signal, taps)
    return y

def hilbert_transform(x: torch.Tensor):
    """
    compute the Hilbert transform
    """
    fourier = torch.fft.fft(x)  #get fft
    h = torch.zeros_like(fourier)
    n = fourier.size(0)
    if n % 2 == 0:      #construct which freqs to zero, which to keep: keep 0 and Nyquist freqs, double positives, zero negatives
        h[0] = h[n//2] = 1  #dc and nyquist
        h[1 : n//2] = 2 #positives
    else:   #odd-length
        h[0] = 1
        h[1:(n+1)//2] = 2
    #apply mask
    z = torch.fft.ifft(fourier * h) #this is the analytical signal
    return z

def gaussian_kernel(sigma: float = 0.2, sr: int = 250, device: str = 'cuda'):
    """
    compute gaussian kernel for filtering later
    """
    radius = int(3 * sigma * sr)    #set gaussian radius
    t = torch.arange(-radius, radius + 1, device=device)    #get tensor
    kernel = torch.exp(-0.5 * (t / (sigma * sr))**2)    #calculate gaussian
    kernel /= kernel.sum()  #normalize
    return kernel.view(1, 1, -1)

def band_powers(signal: torch.Tensor, bands: dict = {'delta': (0.5, 4)}, sr: int = 250, device: str = 'cuda', smoothen = 0.2):
    """
    apply a simple bandpass filter, 
    hilbert transform to return analytical signal, 
    take amplitude and add Gaussian smoothing,
    normalize to 0-1,
    return. 
    """
    signal = signal.to(device) #make sure just in case
    signal = signal.to(dtype = torch.float32)
    if signal.ndim == 1:        #just in case a single channel with [, time] is passed
            signal = signal.unsqueeze(0)
    if signal.size(dim = 0) > 1:    #if there are more channels, takes the first one
        signal = signal[0, :].unsqueeze(0)
    
    band_envelopes = {}
    if smoothen:
        kernel = gaussian_kernel(sigma = smoothen, sr = sr, device = device)
    else:
        kernel = None

    for name, (low, high) in bands.items():
        filtered = bandpass_filter(signal, sr, freqs = (low, high), device=device)
        print('filtered')
        filtered = filtered.squeeze(0)
        analytic = hilbert_transform(filtered)
        amplitude = torch.abs(analytic) 
        print('transformed')
        if amplitude.ndim == 1:
            amplitude = amplitude.unsqueeze(0).unsqueeze(0)
        elif amplitude.ndim == 2:
            amplitude = amplitude.unsqueeze(0)
        
        if kernel is not None:
            amplitude = F.conv1d(amplitude, kernel, padding = 'same').squeeze(0)
        print('convolved')
        q = torch.quantile(amplitude[::10], q=0.95) #every 10th value for speed
        amplitude = torch.clamp(amplitude, max = q) #remove vals above 95th quantile
        #also standardize by std (but not zero-center, because that should happen when filtering)
        amplitude /= torch.std(amplitude)
        amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min() + 1e-12)
        print('done')
        band_envelopes[name] = amplitude
    return band_envelopes   

def sum_power(signal: torch.Tensor, smoothing: float = 0.2, sr: int = 250, device: str = 'cuda', normalize = True):
    """
    calculates sum power by rms with moving average smoothing kernel
    """    
    signal = signal.to(dtype = torch.float32)
    
    if signal is not None:
        if device is not None:
            signal = signal.to(device)
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        out = []
        #here loop over dim 0 - calcualte for all channels
        for ch in range(signal.size(dim = 0)):
            power = (signal[ch, :] ** 2).unsqueeze(0)    #get power by squaring, but keep dim
            win_len = max(1, int(round(smoothing * sr))) #moving average window size
            kernel = torch.ones(1, 1, win_len, device = power.device) / win_len #construct kernel, pytorch requires shape ch_in, ch_out, size. division to get average instead of sum
            pad = win_len // 2  #add some padding
            avg_power = F.conv1d(power.unsqueeze(1), kernel, padding=pad)
            rms = torch.sqrt(avg_power.squeeze(1) + 1e-12)  #very small value ensures float smoothing is never <0
            if normalize:
                rms = (rms - torch.min(rms)) / (torch.max(rms) - torch.min(rms))    #scale
            out.append(rms)
        if len(out) == 1:
            return out[0]
        else:
            return torch.cat(out, dim = 0)
        
def notch_filter(signal,
                 sr:int = 250,
                 freq:int = 50,
                 device:str = 'cuda'):
    """
    designs and applies Notch filter
    creates a Hamming window band-stop filter
    uses fft gpu implementation
    """
    taps = design_notch_hamming(fs = sr, notch_center = freq, notch_width = 2, device = device)
    filtered = fft_filtfilt(signal, taps)
    return filtered