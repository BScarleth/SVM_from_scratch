from Helper import Helper
from MFCC import MFCC
import numpy as np


class AudioTransformMFCC:
    def __init__(self):
        pass

    def __call__(self, sample):
        freq_min = 0
        freq_high = sample['sample_rate'] / 2
        mel_filter_num = 10
        fft_size = 2048
        dct_filter_num = 40

        sample = Helper.normalize_audio(sample)

        frames = MFCC.frame_audio(sample)
        audio_win = MFCC.apply_window(frames)
        audio_fft = MFCC.convert_to_frequency_domain(audio_win)

        audio_power = np.square(np.abs(audio_fft))

        filter_points, mel_freqs = MFCC.get_filter_points(freq_min, freq_high, mel_filter_num, fft_size,
                                                          sample['sample_rate'])

        filter_bank = MFCC.get_filter_bank(filter_points, fft_size)
        filters_norm = MFCC.apply_filter_normalization(mel_freqs, mel_filter_num, filter_bank)
        signal_filtered = MFCC.filter_signal(filters_norm, audio_power)

        dct_filters = MFCC.dct(dct_filter_num, mel_filter_num)
        cepstral_coefficents = np.round(np.dot(dct_filters, signal_filtered),4)

        return {'audio': sample['audio'], 'features': cepstral_coefficents,
                'audio_class': Helper.encode_classes(sample['audio_class'])}
