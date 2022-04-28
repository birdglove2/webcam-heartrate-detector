from scipy.fft import fft, fftfreq

SAMPLE_RATE = 30
DURATION = 30
N = SAMPLE_RATE * DURATION

xf = fftfreq(N, 1 / SAMPLE_RATE)
xf = xf[48:144]
bps_freq=30*xf

print(bps_freq, len(bps_freq))