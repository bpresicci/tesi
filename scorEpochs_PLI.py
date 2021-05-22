"""
 scorEpochs

 Function to select the best (most homogenoous) M/EEG epochs from a
 resting-state recordings.

 INPUT
    cfg: dictionary with the following key-value pairs
         freqRange    - list with the frequency range used to compute the power spectrum (see scipy.stats.spearmanr()
                        function)
         fs           - integer representing sample frequency
         windowL      - integer representing the window length (in seconds)
         smoothFactor - smoothing factor for the power spectrum (0 by default)
         wOverlap     - integer representing the number of seconds of overlap between two consecutive epochs (0 by
                        default)

    data: 2d array with the time-series (channels X time samples)



 OUTPUT

    idx_best_ep: list of indexes sorted according to the best score (this list should be used for the selection of the
                  best epochs)

    epoch:       3d list of the data divided in equal length epochs of length windowL (epochs X channels X time samples)

    score_Xep:   array of score per epoch


 Author:       Simone Maurizio La Cava
 Contributors: Benedetta Presicci





     Copyright (C) 2020 Simone Maurizio La Cava

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import numpy as np
from scipy import signal as sig
from scipy import stats as st
import mne


def scorEpochs_PLI(cfg, data, bands):
    """
    :param cfg:  dictionary with the following key-value pairs
                 freqRange    - list with the frequency range used to compute the power spectrum (see
                                scipy.stats.spearmanr() function)
                 fs           - integer representing sample frequency
                 windowL      - integer representing the window length (in seconds)
                 wOverlap     - integer representing the number of seconds of overlap between two consecutive epochs (0
                                by default)
    :param data: 2d array with the time-series (channels X time samples)

    :return:     idx_best_ep - list of indexes sorted according to the best score (this list should be used for the
                               selection of the best epochs)
                 epoch       - 3d list of the data divided in equal length epochs of length windowL
                               (channels X epochs X time samples)
                 score_Xep   - array of score per epoch
    """

    X = filter_data(data, cfg["fs"], bands)
    epochs, epoch_lenght = split_epoch(X, cfg["fs"], cfg["windowL"])
    X_PLI = np.array([PLI(np.transpose(epoch)) for epoch in epochs])
    n_ch = len(data)
    n_ep = len(epochs)
    X_PLI_x_ch = np.zeros((n_ep, n_ch)) #n_ch == num of computed values with PLI -> X_PLI.shape == (n_ep, n_ch, n_ch)
    score_ch_x_ep = np.zeros((n_ch, n_ep))
    for c in range(n_ch):
        for e in range(n_ep):
            X_PLI_x_ch[e] = X_PLI[e][c]
        score_ch, p = st.spearmanr(X_PLI_x_ch, axis=1)          # Correlation between the PLI of the epochs within each channel
        score_ch_x_ep[c][0:n_ep] += np.mean(score_ch, axis=1)  # Mean similarity score of an epoch with all the epochs for each channel
    score_x_ep = np.mean(score_ch_x_ep, axis=0)                # The score of each epoch is equal to the mean of the scores of all the channels in that epoch
    idx_best_ep = np.argsort(score_x_ep)                     # Obtains of the indexes from the worst epoch to the best
    idx_best_ep = idx_best_ep[::-1]                         # Reversing to obtain the descending order (from the best to the worst)
    return idx_best_ep, epochs, score_x_ep




def _spectrum_parameters(f, freqRange, aux_pxx, nEp, nCh):
    """
    Function which defines the spectrum parameters for the scorEpochs function (FOR INTERNAL USE ONLY).
    """
    idx_min = int(np.argmin(abs(f-freqRange[0])))
    idx_max = int(np.argmin(abs(f-freqRange[-1])))
    nFreq = len(aux_pxx[idx_min:idx_max+1])
    pxx = np.zeros((nEp, nCh, nFreq))
    return pxx, idx_min, idx_max, nFreq

def _overlap(cfg, ep_len, data_len):
    is_overlap = 'wOverlap' in cfg.keys()  # isOverlap = True if the user wants a sliding window; the user will assign the value in seconds of the overlap desired to the key 'wOverlap'
    if is_overlap:
        idx_jump = (cfg['windowL'] - cfg['wOverlap']) * cfg['fs']  # idx_jump is the number of samples that separates the beginning of an epoch and the following one
    else:
        idx_jump = ep_len
    idx_ep = range(0, data_len - ep_len + 1, idx_jump)  # Indexes from which start each epoch
    return idx_ep

def filter_data(raw_data, srate, bands):
    for band in bands:
        low, high = bands[band]
        filtered_data = mne.filter.filter_data(raw_data, srate, low, high)
    return filtered_data

def split_epoch(X, srate, t_epoch_lenght, t_discard=0):
    """
    :param X:  2d array with the time-series EEG data.  n_channels x (duration*srate)
    :param srate: sampling rate
    :param t_epoch_lenght : lenght of the epoch (in seconds)
    :param t_discard: initial portion of the record to be deleted (to eliminate initial artifacts)



    :return:    epoch       - list of the data divided in equal length epochs
                epoch_lenght - lenght in samples - number of samples (per channel) for each epoch


    """

    [n_channels, n_sample] = X.shape

    i_0 = t_discard * srate  # devo eliminare t_discard*srate campioni
    # i_0 non è un tempo - è un indice

    epoch_lenght = t_epoch_lenght * srate  # lunghezza epoca  espressa in numero di campioni

    n_epochs = int((n_sample - i_0 - 1) / epoch_lenght)

    epochs = []
    for i_epoch in range(n_epochs):
        i_start = i_0 + i_epoch * epoch_lenght
        i_stop = i_start + epoch_lenght
        epoch = X[0:n_channels, i_start:i_stop]
        epochs.append(epoch)

    # one-line
    # epochs = [ X[:, i_0 + i_epoch * epoch_lenght: (i_epoch +1 )* epoch_lenght]  for i_epoch in range(n_epochs) ]

    return epochs, epoch_lenght



def PLI(epoch):
    nLoc = np.shape(epoch)[-1]
    PLI = np.zeros(shape=(nLoc, nLoc))
    complex_sig = sig.hilbert(epoch)
    for i in range(nLoc - 1):
        for j in range(i + 1, nLoc):
            PLI[i, j] = abs(np.mean(np.sign(np.angle(np.divide(complex_sig[:, i],complex_sig[:, j])))));
            PLI[j, i] = PLI[i, j];
    return PLI