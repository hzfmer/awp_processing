from scipy.signal import butter, sosfilt, sosfiltfilt
import numpy as np


def filt_B(data_in, fs, lowcut=0, highcut=1, order=4, causal=True):
    if highcut >= fs/2:
        return data_in
    sz = data_in.shape
    if len(sz) > 1 and sz[0] > sz[1]:
        data_in = data_in.T
    data_out = np.zeros_like(data_in)
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    if low == 0:  # lowpass
        sos = butter(order, high, analog=False, btype='low', output='sos')
    else:   # bandpass
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    if len(sz) > 1:
        for i in range(data_in.shape[0]):
            if causal == True:
                data_out[i, :] = sosfilt(sos, data_in[i, :])
                data_out[i, :] = sosfilt(sos, data_out[i, :])
            else:
                data_out[i, :] = sosfiltfilt(sos, data_in[i, :])
        if sz[0] > sz[1]:
            data_out = data_out.T
    else:
        if causal == True:
            data_out = sosfilt(sos, data_in)
            data_out = sosfilt(sos, data_out)
        else:
            data_out = sosfiltfilt(sos, data_in)
    return data_out
