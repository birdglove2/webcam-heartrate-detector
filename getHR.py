import numpy as np
from sklearn.decomposition import FastICA
from scipy.fft import fft, fftfreq
from scipy.fftpack import rfft
from global_vars import FOREHEAD_POINTS, FREQS, INVALID_IDX, WINDOW_SIZE, FPS


def getHeartRate(window):
    # Normalize across the window to have zero-mean and unit variance
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    normalized = (window - mean) / std

    # Separate into three source signals using ICA
    ica = FastICA()
    srcSig = ica.fit_transform(normalized)

    # Find power spectrum
    powerSpec = np.abs(np.fft.fft(srcSig, axis=0)) ** 2
    freqs = np.fft.fftfreq(WINDOW_SIZE, 1.0 / FPS)

    # Find heart rate
    maxPwrSrc = np.max(powerSpec, axis=1)
    validIdx = np.where((freqs >= 50 / 60) & (freqs <= 180 / 60))
    validPwr = maxPwrSrc[validIdx]
    validFreqs = freqs[validIdx]
    maxPwrIdx = np.argmax(validPwr)
    hr = validFreqs[maxPwrIdx]

    return hr
