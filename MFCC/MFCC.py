import numpy as np
from scipy.signal import get_window
import scipy.fftpack as fft
import librosa

class MFCC:
    @staticmethod
    def extract_features(sample, number_of_filters, n_seconds):
        SAMPLE_RATE = 44100
        trim_seconds = SAMPLE_RATE * n_seconds

        data_norm = sample / np.max(np.abs(sample))
        data_norm = data_norm[:trim_seconds]

        mfcc = librosa.feature.mfcc(y=data_norm, sr=SAMPLE_RATE, n_mfcc=number_of_filters)
        return mfcc
