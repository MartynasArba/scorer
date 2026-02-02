import math
import torch
import torch.nn.functional as F

#for filters widmann  et al. 2015 was consulted, settled on FIR filter, Hamming window, FFT implementation for efficiency
def design_bandpass_hamming(fs: int, lowcut: float, highcut: float, 
                            transition_width = 1.0, 
                            numtaps = None, 
                            dtype = torch.float32, 
                            device = 'cuda') -> torch.Tensor:
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
    # if no numtaps given, estimate from transition_width using Hamming approx
    # Hamming window approx transition width Δf ≈ 3.3 / N (normalized to Nyquist units),
    # so N ~ 3.3 / (Δf/nyq) = 3.3 * nyq / Δf
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
    
    # normalize to gain at the passband center frequency
    with torch.no_grad():
        f0 = 0.5 * (lowcut + highcut) # center frequency (Hz)
        w0 = 2.0 * math.pi * f0 / fs  # rad/sample
        n = torch.arange(M, device=device, dtype=dtype)
        # complex frequency response at w0: H(e^{jw0}) = sum h[n] e^{-j w0 n}
        H0 = torch.sum(taps * torch.exp(-1j * w0 * n))
        gain0 = torch.abs(H0).clamp_min(1e-12)
        taps = taps / gain0
        
    #OLD IMPLEMENTATION: normalizing by max
    # normalize by the maximal magnitude of the FFT of taps for better passband scaling).
    # Compute small FFT on CPU to find max magnitude (cheap because numtaps moderate)
    # with torch.no_grad():
    #     # use zero padding to do fine FFT resolution
    #     fftlen = 4096
    #     H = torch.fft.rfft(taps, n=fftlen)
    #     mag = torch.abs(H)
    #     max_mag = float(mag.max())
    #     if max_mag > 0:
    #         taps = taps / max_mag
    return taps  # on device

def design_notch_hamming(fs: int, notch_center: float, 
                         notch_width: float = 2.0, 
                         transition_width: float = 1.0,
                         numtaps: int = None, 
                         dtype = torch.float32, 
                         device: str = 'cuda'):
    """
    designs a linear-phase FIR notch (band-stop) filter using Hamming window 
    something seems broken in the implementation, should be tested
    """
    nyq = fs / 2
    f1 = notch_center - notch_width / 2
    f2 = notch_center + notch_width / 2
    if f1 <= 0 or f2 >= nyq:
        raise ValueError("Notch band must be inside Nyquist range")
    # estimate numtaps if not given
    if numtaps is None:
        numtaps = int(math.ceil(3.3 * nyq / transition_width))
    if numtaps % 2 == 0:
        numtaps += 1

    M = numtaps
    m = torch.arange(M, dtype=dtype, device=device) - (M - 1) / 2

    # convert to radians/sample
    w1 = 2 * math.pi * f1 / fs
    w2 = 2 * math.pi * f2 / fs

    # ideal bandstop = LP(w1) + HP(w2)
    eps = 1e-12
    m_nz = m.clone()
    m_nz[torch.abs(m_nz) < eps] = 1.0

    h_lp_w1 = torch.sin(w1 * m_nz) / (math.pi * m_nz)
    h_lp_w2 = torch.sin(w2 * m_nz) / (math.pi * m_nz)
    h_bs = h_lp_w1 + (torch.zeros_like(h_lp_w1) - h_lp_w2)

    # correct center sample
    c = (M - 1)//2
    h_bs[c] = (w1 - 0) / math.pi + (math.pi - w2) / math.pi

    # Hamming window
    n = torch.arange(M, device=device, dtype=dtype)
    hamming = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (M - 1))
    h_bs = h_bs * hamming
    
    # normalize to DC gain
    dc_gain = torch.sum(h_bs).abs().clamp_min(1e-12)
    h_bs = h_bs / dc_gain
    
    # OLD IMPLEMENTATION: normalize passband gain to max
    # with torch.no_grad():
    #     fftlen = 4096
    #     H = torch.fft.rfft(h_bs, n=fftlen)
    #     max_mag = float(torch.abs(H).max())
    #     h_bs = h_bs / max_mag
    return h_bs

def fft_convolve_1d(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
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

def fft_filtfilt(x: torch.Tensor, taps: torch.Tensor, padlen_factor: int = 3) -> torch.Tensor:
    """
    Zero-phase (forward-reverse) filtering using FFT-based convolution (uses fft_convolve_1d)
    x: (batch, channels, time) or (channels, time) or (time,)
    taps: (numtaps,) symmetric FIR filter
    padlen same as SciPy default
    returns filtered x
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
    
    #instead of conv_func do same padding (old implementation below)
    def fft_convolve_same_1d(x, h):
        # x (..., T), h (K,)
        y_full = fft_convolve_1d(x, h)          # (..., T+K-1)
        K = h.shape[-1]
        start = (K - 1) // 2
        end = start + x.shape[-1]
        return y_full[..., start:end]
    
    y = torch.stack([fft_convolve_same_1d(x_ext[:, i, :], taps) for i in range(C)], dim=1)
    y_rev = torch.flip(y, dims=[-1])
    y2 = torch.stack([fft_convolve_same_1d(y_rev[:, i, :], taps) for i in range(C)], dim=1)
    y_out_ext = torch.flip(y2, dims=[-1])

    y_final = y_out_ext[..., padlen:padlen+T]
    
    # old implementation
    # def conv_func(sig):         #local helper to apply convolution to every channel separately
    #     return fft_convolve_1d(sig, taps)
    # y = torch.stack([conv_func(x_ext[:, i, :]) for i in range(C)], dim=1) # stack convolutions of each channel
    # # reverse to get zero phase
    # y_rev = torch.flip(y, dims=[-1])
    # y2 = torch.stack([conv_func(y_rev[:, i, :]) for i in range(C)], dim=1)
    # y_out_ext = torch.flip(y2, dims=[-1])
    
    # #removing padding
    # start_idx = padlen - (k - 1)
    # end_idx = start_idx + T
    # start_idx = max(start_idx, 0)   
    # end_idx = min(end_idx, y_out_ext.shape[-1])
    # y_final = y_out_ext[..., start_idx:end_idx] #match length to input

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
    joins previous filter design and application funcs
    FIR bandpass filter via FFT convolution on GPU
    signal: [T], [C, T], [batch, channel, time] (should work on all formats)
    hopefully fast!
    """   
    taps = design_bandpass_hamming(sr, freqs[0], freqs[1], transition_width = 1.0, device = device)
    y = fft_filtfilt(signal, taps)
    return y

def hilbert_transform(x: torch.Tensor) -> torch.Tensor:
    """
    compute the Hilbert transform
    returns the analytical signal
    """
    #enorce shape
    if x.ndim == 0:
        raise ValueError("hilbert_transform expected a 1D time series, got a scalar (0D tensor).")
    if x.ndim > 1:
        x = x.reshape(-1)  # flatten
    
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

def band_powers(signal: torch.Tensor, 
                bands: dict = {'delta': (0.5, 4)}, 
                sr: int = 250, 
                device: str = 'cuda', 
                smoothen: float = 0.2,
                scale = True) -> dict[torch.Tensor]:
    """
    apply a simple bandpass filter, 
    hilbert transform to return analytical signal, 
    take amplitude and add Gaussian smoothing (optional),
    scale,
    return band powers
    """
    signal = signal.to(device) #make sure just in case
    signal = signal.to(dtype = torch.float32)
    if signal.ndim == 1:        #just in case a single channel with [, time] is passed
            signal = signal.unsqueeze(0)
    if signal.size(dim = 0) > 1:    #if there are more channels, takes the first one
        print('multiple channels passed for band powers, using channel 0')
        signal = signal[0, :]   #should be 1, time
    
    band_envelopes = {}
    if smoothen:
        kernel = gaussian_kernel(sigma = smoothen, sr = sr, device = device)
    else:
        kernel = None

    for name, (low, high) in bands.items():
        filtered = bandpass_filter(signal, sr, freqs = (low, high), device=device)  #still 1, time
        #enforce 1d for Hilbert transform
        if filtered.ndim == 2:          # 1, T or C, T
            filtered1d = filtered[0]    # take channel 0
        elif filtered.ndim == 1:     
            filtered1d = filtered
        else:
            raise ValueError(f"Unexpected filtered shape: {filtered.shape}")
       
        analytic = hilbert_transform(filtered1d)        
        amplitude = torch.abs(analytic) 
        
        #expand for convolution to 1, 1, time
        amplitude = amplitude.unsqueeze(0).unsqueeze(0)
        
        if kernel is not None:
            amplitude = F.conv1d(amplitude, kernel, padding = 'same').squeeze(0)
        
        #force 1, time always - reduce to single dim, then expand
        amplitude = amplitude.squeeze(0).squeeze(0).unsqueeze(0)
        
        eps = 1e-20
        sub = amplitude[0, ::10]
        q = torch.quantile(sub, q=0.98) #every 10th value for speed
        q25 = torch.quantile(sub, q=0.25)
        q75 = torch.quantile(sub, q=0.75)
        amplitude = torch.clamp(amplitude, max = q) #remove vals above 90th quantile
        #also standardize by std (but not zero-center, because that should happen when filtering)
        # amplitude /= torch.std(amplitude)
        # amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min() + 1e-12)
        #robust scaling
        if scale:
            amplitude = (amplitude - q25) / (q75 - q25 + eps)
            amplitude = torch.clamp(amplitude, min = -2, max = 6)        
        band_envelopes[name] = amplitude
        
    return band_envelopes   

def sum_power(signal: torch.Tensor, 
              smoothing: float = 0.2, 
              sr: int = 250, 
              device: str = 'cuda', 
              normalize: bool = True, 
              gaussian_smoothen:float = 0.2) -> torch.Tensor:
    """
    calculates sum power by rms with moving average smoothing kernel
    """    
    signal = signal.to(dtype = torch.float32)
    
    if signal is not None:
        if device is not None:
            signal = signal.to(device)
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
            
        if gaussian_smoothen is not None:
            smoothing_kernel = gaussian_kernel(sigma = gaussian_smoothen, sr = sr, device = device)
        else:
            smoothing_kernel = None
        
        out = []
        #here loop over dim 0 - calcualte for all channels
        for ch in range(signal.size(dim = 0)):
            power = (signal[ch, :] ** 2).unsqueeze(0)    #get power by squaring, but keep dim
            win_len = max(1, int(round(smoothing * sr))) #moving average window size
            if win_len % 2 == 0:
                win_len += 1   # force odd length
            kernel = torch.ones(1, 1, win_len, device = power.device) / win_len #construct kernel, pytorch requires shape ch_in, ch_out, size. division to get average instead of sum
            # pad = win_len // 2  #add some padding #padding here will result in uneven window length if even, so either force odd or use "same" padding
            avg_power = F.conv1d(power.unsqueeze(1), kernel, padding = "same")  #can use "same" padding as I don't need stride != 1
            rms = torch.sqrt(avg_power.squeeze(1) + 1e-12)  #very small value ensures float smoothing is never <0
            
            #add gaussian smoothing if set
            if smoothing_kernel is not None:
                if rms.ndim == 2:
                    rms = rms.unsqueeze(1)
                elif rms.ndim == 1:
                    rms = rms.unsqueeze(0).unsqueeze(1)  # (1, 1, T)

                rms = F.conv1d(rms, smoothing_kernel, padding='same')  # (1, 1, T)
                rms = rms.squeeze(1)  # (1, T)   (keeps 2D)
            
            if normalize:
                eps = 1e-20
                sub = rms[..., ::10]    #assuming shape 1 (channel), time
                q = torch.quantile(sub, q=0.98) #every 10th value for speed
                q25 = torch.quantile(sub, q=0.25)
                q75 = torch.quantile(sub, q=0.75)
                rms = torch.clamp(rms, max = q) #remove vals above 90th quantile                
                # rms = (rms - torch.min(rms)) / (torch.max(rms) - torch.min(rms))    #scale
                #use more robust scaling instead of minmax
                rms = (rms - q25) / (q75 - q25 + eps)
                rms = torch.clamp(rms, min = 0, max = 2)
                
            if rms.ndim == 1:
                rms = rms.unsqueeze(0)  # make (1(channel), T)
            out.append(rms)
            
        if len(out) == 1:
            return out[0]
        else:
            return torch.cat(out, dim = 0)
        
def notch_filter(signal,
                 sr:int = 250,
                 freq:int = 50,
                 device:str = 'cuda') -> torch.Tensor:
    """
    designs and applies Notch filter
    creates a Hamming window band-stop filter
    uses fft gpu implementation
    """
    taps = design_notch_hamming(fs = sr, notch_center = freq, notch_width = 2, device = device)
    filtered = fft_filtfilt(signal, taps)
    return filtered

def preprocess_test(sr: int = 250, 
                    T: float = 20.0, 
                    device: str = "cuda"):
    """function to test filters"""
    import matplotlib.pyplot as plt
    
    def _to_1d(x: torch.Tensor) -> torch.Tensor:
        """helper, enforces 1d shape"""
        if x.ndim == 2 and x.shape[0] == 1:
            return x[0]
        if x.ndim != 1:
            return x.reshape(-1)
        return x

    torch.set_default_dtype(torch.float32)
    N = int(sr * T)
    t = torch.arange(N, device=device) / sr

    # 1) IMPULSE TEST (timing / filtfilt symmetry)
    x_imp = torch.zeros(N, device=device)   #generate signal
    mid = N // 2
    x_imp[mid] = 1.0
    y_imp = bandpass_filter(x_imp, sr=sr, freqs=(0.5, 4.0), device=device)
    y_imp = _to_1d(y_imp)
    #peak should remain at the same index if zero-phase is correct
    peak_idx = int(torch.argmax(torch.abs(y_imp)).item())
    print(f"[Impulse] peak at {peak_idx}, expected {mid}, shift = {peak_idx - mid} samples")
    
    # 2) SINE BURST TEST (envelope alignment)
    # Create a burst of 10 Hz in the middle (Gaussian-windowed)
    f0 = 10.0
    sigma_s = 0.5  # seconds, controls burst width
    env = torch.exp(-0.5 * ((t - T/2) / sigma_s) ** 2)
    x_burst = env * torch.sin(2 * math.pi * f0 * t)
    # Filter around alpha-ish band containing 10 Hz
    y_burst = bandpass_filter(x_burst, sr=sr, freqs=(8.0, 12.0), device=device)
    y_burst = _to_1d(y_burst)
    # Envelope from Hilbert
    a = torch.abs(hilbert_transform(y_burst))
    env_peak = int(torch.argmax(env).item())
    a_peak = int(torch.argmax(a).item())
    print(f"[Burst] true env peak at {env_peak}, Hilbert amp peak at {a_peak}, shift = {a_peak - env_peak} samples")

    # 3) BAND ENVELOPES vs RMS TEST (rough consistency)
    # Multi-tone + noise
    x_mix = (
        0.8 * torch.sin(2 * math.pi * 2.0 * t) +   # delta-ish
        0.5 * torch.sin(2 * math.pi * 10.0 * t) +  # alpha-ish
        0.3 * torch.sin(2 * math.pi * 30.0 * t) +  # beta-ish
        0.1 * torch.randn_like(t)
    )

    bands = {
        "delta": (0.5, 4.0),
        "alpha": (8.0, 12.0),
        "beta":  (13.0, 35.0),
    }

    bp = band_powers(signal=x_mix, bands=bands, sr=sr, device=device, smoothen=0.2, scale = False)
    rms = sum_power(signal=x_mix.unsqueeze(0), sr=sr, device=device, smoothing=0.2, gaussian_smoothen=0.2, normalize=False)
    rms = rms[0] 
    
    # Sum of (UN-normalized) band envelopes should correlate with overall RMS trends.
    # band_powers currently robust-scales/clamps, so correlation is qualitative.
    band_sum = torch.zeros_like(rms)
    for k in bp:
        band_sum += _to_1d(bp[k]).squeeze(0)

    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    axs[0].plot(x_imp.cpu().numpy(), label="impulse")
    axs[0].plot(y_imp.cpu().numpy(), label="bandpass(filtfilt)")
    axs[0].axvline(mid, linestyle="--")
    axs[0].axvline(peak_idx, linestyle=":")
    axs[0].set_title("Impulse response check (zero-phase alignment)")
    axs[0].legend()

    axs[1].plot(x_burst.cpu().numpy(), label="burst signal")
    axs[1].plot(env.cpu().numpy(), label="true envelope")
    axs[1].plot(a.cpu().numpy(), label="Hilbert amplitude (after bandpass)")
    axs[1].axvline(env_peak, linestyle="--")
    axs[1].axvline(a_peak, linestyle=":")
    axs[1].set_title("Envelope alignment check")
    axs[1].legend()

    axs[2].plot(rms.detach().cpu().numpy(), label="RMS (not normalized)")
    axs[2].plot(band_sum.detach().cpu().numpy(), label="sum(band envelopes) (scaled/clamped)")
    axs[2].set_title("Band envelopes vs RMS (qualitative consistency)")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print('testing...')
    preprocess_test()