from scipy import signal as sig
import numpy as np
from scipy import fft

def feature_extraction(cfg, freq_range, nCh, epoch, idx_ep):
    epLen = cfg['windowL'] * cfg['fs']  # Number of samples of each epoch (for each channel)
    smoothing_condition = 'smoothFactor' in cfg.keys() and cfg['smoothFactor'] > 1  # True if the smoothing has to be executed, 0 otherwise
    nEp = len(idx_ep)  # Total number of epochs

    segLen = round(epLen/8)
    check_freqs = fft.rfftfreq(segLen, 1/cfg['fs'])
    check = 0
    for value, i in zip(check_freqs, range(len(check_freqs))):
      if value >= cfg['freqRange'][0] and value <= cfg['freqRange'][1]:
        check += 1
    if check < 9:
      fix = (cfg['freqRange'][1] - cfg['freqRange'][0]) / 9
      segLen = 1.0/(fix * 1/cfg['fs'])

    for e in range(nEp):
        for c in range(nCh):
            # compute power spectrum
            f, aux_pxx = sig.welch(epoch[e][c].T, cfg['fs'], window='hamming', nperseg=round(epLen / 8), detrend=False)  # The nperseg allows the MATLAB pwelch correspondence
            if c == 0 and e == 0:  # The various parameters are obtained in the first interation
                psd, idx_min, idx_max, nFreq = _spectrum_parameters(f, freq_range, aux_pxx, nEp, nCh)
                if smoothing_condition:
                    window_range, initial_f, final_f = _smoothing_parameters(cfg['smoothFactor'], nFreq)
            if smoothing_condition:
                psd[e][c] = _movmean(aux_pxx, cfg['smoothFactor'], initial_f, final_f, nFreq, idx_min, idx_max)
            else:
                psd[e][c] = aux_pxx[idx_min:idx_max + 1]  # pxx takes the only interested spectrum-related sub-array
    return psd

def feature_extraction_1(cfg, freq_range, nCh, epoch):
    epLen = cfg['windowL'] * cfg['fs']  # Number of samples of each epoch (for each channel)
    smoothing_condition = 'smoothFactor' in cfg.keys() and cfg['smoothFactor'] > 1  # True if the smoothing has to be executed, 0 otherwise
    nEp = len(epoch)  # Total number of epochs

    segLen = round(epLen/8)
    check_freqs = fft.rfftfreq(segLen, 1/cfg['fs'])
    check = 0
    for value, i in zip(check_freqs, range(len(check_freqs))):
      if value >= cfg['freqRange'][0] and value <= cfg['freqRange'][1]:
        check += 1
    if check < 9:
      fix = (cfg['freqRange'][1] - cfg['freqRange'][0]) / 9
      segLen = 1.0/(fix * 1/cfg['fs'])

    for e in range(nEp):
        for c in range(nCh):
            # compute power spectrum
            f, aux_pxx = sig.welch(epoch[e][c].T, cfg['fs'], window='hamming', nperseg=segLen, detrend=False)  # The nperseg allows the MATLAB pwelch correspondence
            if c == 0 and e == 0:  # The various parameters are obtained in the first interation
                psd, idx_min, idx_max, nFreq = _spectrum_parameters(f, freq_range, aux_pxx, nEp, nCh)
                if smoothing_condition:
                    window_range, initial_f, final_f = _smoothing_parameters(cfg['smoothFactor'], nFreq)
            if smoothing_condition:
                psd[e][c] = _movmean(aux_pxx, cfg['smoothFactor'], initial_f, final_f, nFreq, idx_min, idx_max)
            else:
                psd[e][c] = aux_pxx[idx_min:idx_max + 1]  # pxx takes the only interested spectrum-related sub-array
    return psd

def _movmean(aux_pxx, smoothFactor, initial_f, final_f, nFreq, idx_min, idx_max):   #It is not weighted
    """
    Function used for computing the smoothed power spectrum through moving average filter, where each output sample is
    evaluated on the center of the window at each iteration (or the one furthest to the right of the two in the center,
    in case of a window with an even number of elements), without padding on edges (FOR INTERNAL USE ONLY).
     X = [X(0), X(1), X(2), X(3), X(4)]
     smoothFactor = 3
     Y(0) = (X(0))+X(1))/2
     Y(1) = (X(0)+X(1)+X(2))/3
     Y(2) = (X(1)+X(2)+X(3))/3
     Y(3) = (X(2)+X(3)+X(4))/3
     Y(4) = (X(3)+X(4))/2
    """
    smoothed = np.zeros((idx_max-idx_min+1,))
    for f in range(nFreq):
        if f < initial_f:
            smoothed[f] = np.mean(aux_pxx[idx_min:idx_min+f+initial_f+1])
        elif f >= final_f:
            smoothed[f] = np.mean(aux_pxx[idx_min+f-initial_f:])
        elif smoothFactor % 2 == 0:
            smoothed[f] = np.mean(aux_pxx[idx_min+f-initial_f:idx_min+f+initial_f])
        else:
            smoothed[f] = np.mean(aux_pxx[idx_min+f-initial_f:idx_min+f+initial_f+1])
    return smoothed


def _spectrum_parameters(f, freqRange, aux_pxx, nEp, nCh):
    """
    Function which defines the spectrum parameters for the scorEpochs function (FOR INTERNAL USE ONLY).
    """
    idx_min = int(np.argmin(abs(f - freqRange[0])))
    idx_max = int(np.argmin(abs(f - freqRange[-1])))
    nFreq = len(aux_pxx[idx_min:idx_max + 1])
    pxx = np.zeros((nEp, nCh, nFreq))
    return pxx, idx_min, idx_max, nFreq


def _smoothing_parameters(smoothFactor, nFreq):
    """
    Function which defines the parameters of the window to be used by the scorEpochs function in smoothing the spectrum
    (FOR INTERNAL USE ONLY).
    """
    window_range = round(smoothFactor)
    initial_f = int(window_range/2)
    final_f = nFreq - initial_f
    return window_range, initial_f, final_f

