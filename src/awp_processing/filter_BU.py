from scipy.signal import butter, sosfilt, sosfiltfilt
import numpy as np

def filt_B(data_in, fs, lowcut=0, highcut=1, order=4, causal=True, axis=0):
    """ Butterworth filter
    Input
    -----
    date_in : 1D or 2D array
    fs : float
        Sampling frequency of data_in
    lowcut : float
        Lower frequency boundary, low-pass filtering if 0
    highcut : float
        Higher frequency boudanry; high-pass not explictly implemented
    order : int 
    causal : bool
        If True, do forward-backward filtering (zero-phase); otherwise two-pass forward filtering (acausal signals before initial)
    axis : int
        If set to 1, force the last axis to be the time axis

    Return
    ------
    data_out : shape of data_in
        Filtered data, with the same shape as data_in
    """
    if highcut >= fs/2:
        print("Upper band frequency larger than Nyquist, no filtering is done!")
        return data_in
    sz = data_in.shape
    if len(sz) > 1 and sz[0] > sz[1] and not axis:
        # Make sure the time axis is the last axis
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
        if sz[0] > sz[1] and not axis:
            data_out = data_out.T
    else:
        if causal == True:
            data_out = sosfilt(sos, data_in)
            data_out = sosfilt(sos, data_out)
        else:
            data_out = sosfiltfilt(sos, data_in)
    return data_out
