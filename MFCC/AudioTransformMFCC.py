from MFCC.MFCC import MFCC

class AudioTransformMFCC:
    def __init__(self, number_of_filters = 13, number_of_seconds = 10):
        self.number_of_filters = number_of_filters
        self.number_of_seconds = number_of_seconds


    def __call__(self, sample):
        features = MFCC.extract_features(sample['audio'], self.number_of_filters, self.number_of_seconds)

        return {'features': features, 'audio_class': sample['audio_class']}
