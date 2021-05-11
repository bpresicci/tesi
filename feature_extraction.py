from scipy import signal as sig
import numpy as np
from scipy import fft

def feature_extraction_1(cfg, freq_range, nCh, epoch):
    epLen = cfg['windowL'] * cfg['fs']  # Number of samples of each epoch (for each channel)
    smoothing_condition = 'smoothFactor' in cfg.keys() and cfg['smoothFactor'] > 1  # True if the smoothing has to be executed, 0 otherwise
    nEp = len(epoch)  # Total number of epochs

    segLen = epLen/8
    check_freqs = fft.rfftfreq(segLen, 1/cfg['fs'])
    check = 0
    for value, i in zip(check_freqs, range(len(check_freqs))):
      if value >= cfg['freqRange'][0] and value <= cfg['freqRange'][1]:
        check += 1
    if check < 9:
      fix = (cfg['freqRange'][1] - cfg['freqRange'][0]) / 9
      segLen = 1.0/(fix * 1/cfg['fs'])

# L'array di frequenze restituite da sig.welch() ("Array of sample frequencies") dipende dal parametro nperseg, che è la lunghezza di ogni segmento. Volendo ottenere un numero
# sufficiente di valori di PSD da usare per ricavare i coefficienti di Spearman, mantenendo una finestra di osservazione (in secondi, cfg['windowL'] piccola
# ed una ridotta banda di frequenza (per esempio la banda alpha, [8, 13]) si deve modificare il calcolo di nperseg.
# La funzione chiamata da sig.welch() per ottenere le frequenze è fft.rfftfreq() di scipy, ma numpy ha una funzione identica che fa gli stessi calcoli.
# Il codice sorgente di numpy.fft.rfftfreq() (https://github.com/numpy/numpy/blob/v1.20.0/numpy/fft/helper.py#L172-L221) comprende:

# Parameters
#    ----------
#    n : int
#       Window length.
#   d : scalar, optional
#       Sample spacing (inverse of the sampling rate). Defaults to 1.
#   Returns
#   -------
#   f : ndarray
#       Array of length ``n//2 + 1`` containing the sample frequencies.

# def rfftfreq(n, d=1.0):
#   if not isinstance(n, integer_types):
#       raise ValueError("n should be an integer")
#   val = 1.0/(n*d)
#   N = n//2 + 1
#   results = arange(0, N, dtype=int)
#   return results * val

# Sia cfg['freqRange'] = [low, high], se si vuole imporre che in tale banda vi siano almeno 9 frequenze associate ad altrettanti valori di PSD, si deve modificare
# il valore di val, perchè è la distanza fra una frequenza e quella successiva nell'array f restituito in output da sig.welch()
# (si può verificare che f[1] - f[0] = val). d è l'inverso di fs, n è nperseg, quindi:
# val = 1/(n*d) = fs/nperseg
# numero di frequenze in cfg['freqRange'] = (high - low) / val
# Quindi:
# (high - low) / val >= 9
# val <= (high - low) / 9
# da cui:
# fix = (cfg['freqRange'][1] - cfg['freqRange'][0]) / 9
# val = fs / nperseg -> nperseg = fs/val = 1/(val * d) con d = 1/fs
# da cui: 
# segLen = 1.0/(fix * 1/cfg['fs'])
# Così si ottengono un array f e pxx da sig.welch() la cui lunghezza nell'intervallo definito da cfg['freqRange'] è almeno 9.

    for e in range(nEp):
        for c in range(nCh):
            # compute power spectrum
            f, aux_pxx = sig.welch(epoch[e][c].T, cfg['fs'], window='hamming', nperseg=round(segLen), detrend=False)  # The nperseg allows the MATLAB pwelch correspondence
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

