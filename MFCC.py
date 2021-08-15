import numpy as np
from scipy.signal import get_window
import scipy.fftpack as fft


class MFCC:
    """
    FFT_size = The window size must be a power of 2, and defaults to 512.
    The FFT size defines the number of bins used for dividing the window into equal strips, or bins.
    """

    @staticmethod
    def frame_audio(audio_sample, fft_size=2048, overlap_factor=10):
        audio_sample['audio'] = np.pad(audio_sample['audio'], int(fft_size / 2), mode='reflect')
        frame_len = np.round(audio_sample['sample_rate'] * overlap_factor / 1000).astype(int)
        frame_num = int((len(audio_sample['audio']) - fft_size) / frame_len) + 1
        frames = np.zeros((frame_num, fft_size))

        for n in range(frame_num):
            frames[n] = audio_sample['audio'][n * frame_len:n * frame_len + fft_size]

        return frames

    @staticmethod
    def apply_window(audio_framed, fft_size=2048):
        window = get_window("hann", fft_size, fftbins=True)
        return audio_framed * window

    @staticmethod
    def convert_to_frequency_domain(audio_with_window, fft_size=2048):
        audio_winT = np.transpose(audio_with_window)
        audio_fft = np.empty((int(1 + fft_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')

        for n in range(audio_fft.shape[1]):
            audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
        return np.transpose(audio_fft)

    @staticmethod
    def freq_to_mel(freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    @staticmethod
    def met_to_freq(mels):
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    @staticmethod
    def get_filter_points(fmin, fmax, mel_filter_num, fft_size, sample_rate):
        fmin_mel = MFCC.freq_to_mel(fmin)
        fmax_mel = MFCC.freq_to_mel(fmax)

        mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num + 2)
        freqs = MFCC.met_to_freq(mels)

        return np.floor((fft_size + 1) / sample_rate * freqs).astype(int), freqs

    @staticmethod
    def get_filter_bank(filter_points, fft_size):
        filters = np.zeros((len(filter_points) - 2, int(fft_size / 2 + 1)))

        for n in range(len(filter_points) - 2):
            filters[n, filter_points[n]: filter_points[n + 1]] = np.linspace(0, 1,
                                                                             filter_points[n + 1] - filter_points[n])
            filters[n, filter_points[n + 1]: filter_points[n + 2]] = np.linspace(1, 0,
                                                                                 filter_points[n + 2] - filter_points[
                                                                                     n + 1])
        return filters

    @staticmethod
    def apply_filter_normalization(mel_freqs, mel_filter_num, filters):
        enorm = 2.0 / (mel_freqs[2:mel_filter_num + 2] - mel_freqs[:mel_filter_num])
        filters *= enorm[:, np.newaxis]
        return filters

    @staticmethod
    def filter_signal(filters, audio_power):
        audio_filtered = np.dot(filters, np.transpose(audio_power))
        audio_log = 10.0 * np.log10(audio_filtered)
        return audio_log

    @staticmethod
    def dct(dct_filter_num, filter_len):
        basis = np.empty((dct_filter_num, filter_len))
        basis[0, :] = 1.0 / np.sqrt(filter_len)

        samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

        for i in range(1, dct_filter_num):
            basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)

        return basis
