import numpy as np
from scipy.fftpack import dct
from sidekit.frontend.features import audspec, postaud, dolpc, lifter
from .utils import *
  
class BaseProcessor:
    def __init__(self, sr):
        self.sr = sr

    def __call__(self, signal):
        pass


class Frame(BaseProcessor):
    def __init__(self, sr, frame_sec, overlap):
        super().__init__(sr)
        self.frame_len = int(sr*frame_sec)
        self.hop_len = int((1-overlap)*self.frame_len)

    def __call__(self, signal):
        idx = 0
        n = len(signal)
        frames = []
        while idx < n:
            frame = signal[idx:idx+self.frame_len]
            frames += [np.pad(frame, (0, self.frame_len - len(frame)))]
            idx += self.hop_len
        return np.array(frames)

    
class Spectrogram(BaseProcessor):
    def __init__(self, sr, n_fft, window_type):
        super().__init__(sr)
        self.n_fft = n_fft
        self.window = hanning(n_fft) if window_type == 'hann' else hamming(n_fft)

    def __call__(self, sig):
        framed = preemphasis(sig)
#         sig = np.pad(sig, (0, self.n_fft - sig.shape[1]))
        return np.square(np.abs(np.fft.rfft(sig*self.window, n=self.n_fft))).T / self.n_fft


class MFCC(BaseProcessor):
    def __init__(self, sr, n_fft, window_func, n_mel, n_cep):
        super().__init__(sr)
        self.n_fft = n_fft
        self.filters = mel_filters(sr, n_fft, n_mel)
        self.n_cep = n_cep

    def __call__(self, spec):
        mel_spec = np.dot(spec.T, self.filters.T)
        mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)  # Numerical Stability
        mel_spec = np.log(mel_spec)
        return dct(mel_spec, type=2, axis=1, norm='ortho')[:,:self.n_cep].T


class PLP(BaseProcessor):
    def __init__(self, sr, order, n_bark):
        super().__init__(sr)
        self.order = order
        self.n_bark = n_bark

    def __call__(self, spec):
        # next group to critical bands
        audio_spectrum = audspec(spec.T, fs=self.sr, nfilts=self.n_bark, fbtype='bark', minfreq=0, maxfreq=self.sr/2)[0]
        nbands = audio_spectrum.shape[0]
        # do final auditory compressions
        post_spectrum = postaud(audio_spectrum, fmax=self.sr / 2, fbtype='bark')[0]

        # LPC analysis
        lpcas = dolpc(post_spectrum, self.order - 1)
        # convert lpc to cepstra
        cepstra = lpc2cep(lpcas.T, self.order)
        # cepstra = lifter(cepstra, 0.6)

        return cepstra



class LPC(BaseProcessor):
    def __init__(self, sr, order):
        super().__init__(sr)
        self.order = order
    
    def __call__(self, spec):
        return dolpc(spec.T, self.order-1).T


class Pipeline:
    def __init__(self, processors):
        assert isinstance(processors, list)
        self.processors = processors

    def __call__(self, signal):
        res = []
        y = signal
        for processor in self.processors:
            y = processor(y)
            res += [(str(type(processor)), y)]
        return y, res