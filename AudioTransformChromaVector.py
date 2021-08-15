from Helper import Helper
from MFCC import MFCC
import numpy as np

class AudioTransformMFCC:
    def __init__(self):
        pass

    def __call__(self, sample):
        chroma_vector = []

        return {'audio': sample['audio'], 'sample_rate': sample['sample_rate'],
                'features': chroma_vector, 'audio_class': Helper.encode_classes(sample['audio_class'])}
