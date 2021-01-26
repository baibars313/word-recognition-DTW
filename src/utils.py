import librosa
import numpy as np

def read_audio(file, sr=16000):
    x, fs = librosa.load(file, sr=sr)
    return x

def preemphasis(signal, coeff=0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def hamming(n):
    x = np.arange(n)
    return 0.54 - 0.46*np.cos(2*np.pi*x/(n-1))

def hanning(n):
    x = np.arange(n)
    return 0.5 - 0.5*np.cos(2*np.pi*x/(n-1))

def fft_frequencies(n_fft=480, sr=16000):
    return np.linspace(0, sr/2, n_fft // 2 + 1)

def hz2mel(freq):
    return 2595 * np.log10(1+freq/700.)

def mel2hz(freq):
    return 700.0 * (10.0**(freq / 2595.0) - 1.0)

def hz2bark(freq):
    return 6. * np.arcsinh(freq / 600.)

def bark2hz(freq):
    return 600. * np.sinh(freq / 6.)


def mel_filters(sr, n_fft, n_filters):
    low = hz2mel(0.)
    high = hz2mel(sr/2)
    mels = np.linspace(low, high, n_filters+2)
    fft_bins = np.floor((n_fft+1) * mel2hz(mels) / sr)
    filters = np.zeros([n_filters, n_fft//2 + 1])
    for j in range(0, n_filters):
        for i in range(int(fft_bins[j]), int(fft_bins[j+1])):
            filters[j,i] = (i - fft_bins[j]) / (fft_bins[j+1]-fft_bins[j])
        for i in range(int(fft_bins[j+1]), int(fft_bins[j+2])):
            filters[j,i] = (fft_bins[j+2]-i) / (fft_bins[j+2]-fft_bins[j+1])
    return filters

def lpc2cep(lpc, n_cep):
    m, n = lpc.shape
    order = m
    n_cep = min(n_cep, order)
    cep = np.zeros((n_cep, n))
    cep[0, :] = -np.log(lpc[0, :])
    lpc_norm = lpc/(np.tile(lpc[0, :], (m, 1)) + 1e-8)

    for i in range(1, n_cep):
        s = 0
        for j in range(1, i):
            s += ((i-j)*lpc_norm[j, :])*cep[(i-j), :]
        cep[i, :] = -(lpc_norm[i, :]+ s/i)

    return cep