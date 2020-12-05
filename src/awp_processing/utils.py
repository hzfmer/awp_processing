class AttrDict(dict):
    """ Nested Attribute Dictionary

    A class to convert a nested Dictionary into an object with key-values
    accessible using attribute notation (AttrDict.attribute) in addition to
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse into nested dicts (like: AttrDict.attr.attr)
    """

    def __init__(self, mapping=None):
        #super(AttrDict, self).__init__()
        super().__init__()
        if mapping is not None:
            for key, value in mapping.items():
                self.__setitem__(key.lower() if type(key) == str else key, value)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        super().__setitem__(key, value)
        self.__dict__[key] = value  # for code completion in editors

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    __setattr__ = __setitem__



from collections.abc import Iterable

import numpy as np


def check_iterable(x):
    return isinstance(x, Iterable) or isinstance(x, str)

    
def force_iterable(x):
    return x if check_iterable(x) else [x]


def save_image(fig):
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

def decimate(arr, limit=1500, count=800):
    """Decimate snapshots to save memory
    Return
    ------
        [int] steps to skip
    """
    if len(arr) > limit:
        return len(arr) // count
    return 1

def distance(lon1, lat1, lon2, lat2):
    lat1, lon1, lat2, lon2 = np.radians((lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    d = 0.5 - np.cos(dlat) / 2 + np.cos(lat1) * np.cos(lat2)  * (1 - np.cos(dlon)) / 2
    return 12742 * np.arcsin(np.sqrt(d))


def resize(data1, dt1, data2, dt2, dt):
    """Trim time series and resample
    Input
    -----
        data1, data2 : list of float
            Time series
        dt1, dt2 : float
            Time steps of data1 and data2
        dt : float
            Time step after resampling

    Return
    ------
        data1, data2 : list of float
            Trimmed and resampled time series
    """
        
    from scipy.signal import resample
    if dt1 != dt:
        data1 = resample(data1, int(len(data1) * dt1 // dt))
    if dt2 != dt:
        data2 = resample(data2, int(len(data2) * dt2 // dt))
    if len(data1) > len(data2):
        data1 = data1[:len(data2)]
    return data1, data2

            
def comp_fft(data, dt, fmax=None):
    """Compute fft of data
    Input
    -----
        data : list or 2D array of floats
            Input time series
        dt : float
            Time step
        fmax : float or None
            Maximum frequency, using half Nyquist if not specified
    Return 
    ------
        f : list of float 
            frequency list
        fourier : the same type as data
            Output Fourier spectra
    """
    if len(data.shape) > 2:
        print(f"Deal with 1D or 2D array, given {len(data.shape)}.\nAborting!")
        return 
    if len(data.shape) == 1:
        data = data[:, None]
    if data.shape[0] > data.shape[1]:
        # Make sure last dimension longest
        f, fourier = comp_fft(data.T, dt, fmax=fmax)
        return f, fourier.T
    length = data.shape[-1]
    fourier = np.abs(np.fft.fft(data, axis=-1) * 2) * dt

    fs = 1 / dt
    df = fs / length
    f = np.arange(length) * df + df
    if not fmax:
        fmax = fs / 2
    else:
        if fmax > fs / 2:
            print(f"Cap fmax ({fmax:.2f}) at half Nyquist ({fs / 2:.2f})")
        fmax = min(fmax, fs / 2)
    fourier = fourier[:, f<=fmax]
    f = f[f <= fmax]
    return fourier.squeeze(), f

import my_pyrotd


def comp_psa(v=dict(), dt=0, vx=None, vy=None, percentiles=[50], osc_freqs=np.logspace(-1, 1, 91), osc_damping=0.05):
    """Compute pseudospectrum"""
    # RotD50 for percentiles=[50] 
    if not dt and 'dt' not in v:
        print("No dt provided! Abort")
        return None
    dt = dt or v['dt']
    vx = vx or v['x']
    vy = vy or v['y']
    accx = np.gradient(vx, dt)
    accy = np.gradient(vy, dt)
    # with keys 'osc_freq', 'percentile', 'spec_accel'
    return my_pyrotd.my_calc_rotated_spec_accels(
            dt, accx, accy, osc_freqs,
            osc_damping, percentiles=percentiles)
