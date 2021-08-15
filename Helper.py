import numpy as np
import matplotlib.pyplot as plt
import json


class Helper:

    @staticmethod
    def read_file(filename):
        lines = []
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                lines.append(line)
        return lines

    @staticmethod
    def normalize_audio(audio_sample):
        audio_sample['audio'] = audio_sample['audio'] / np.max(np.abs(audio_sample['audio']))
        audio_sample['audio'] = audio_sample['audio'][:882000]
        return audio_sample

    @staticmethod
    def show_audio(audio_sample, title):
        plt.figure(figsize=(15, 4))
        plt.plot(np.linspace(0, len(audio_sample['audio']) / audio_sample['sample'],
                             num=len(audio_sample['audio'])), audio_sample['audio'])
        plt.grid(True)
        plt.title(title)
        plt.show()

    @staticmethod
    def show_spectrum(sample):
        plt.figure(figsize=(15, 5))
        plt.plot(np.linspace(0, len(sample["audio"]) / sample["sample_rate"], num=len(sample["audio"])),
                 sample["audio"])
        plt.imshow(sample["c-coefficients"], aspect='auto', origin='lower')
        title = "Example of features from: "+ str(sample["audio_class"]) + " language"
        plt.title(title)
        plt.show()

    @staticmethod
    def encode_classes(class_name):
        if class_name == "arabic":
            return 1
        if class_name == "dutch":
            return 2
        if class_name == "english_australia":
            return 3
        if class_name == "english_usa":
            return 4
        if class_name == "french":
            return 5
        if class_name == "german":
            return 6
        if class_name == "korean":
            return 7
        if class_name == "russian":
            return 8
        if class_name == "spanish":
            return 9
        if class_name == "turkish":
            return 10

    @staticmethod
    def get_class_from_file_name(filename):
        return filename.split('/')[0]

    @staticmethod
    def read_features_file(filename):
        audios_dict = []

        with open(filename + ".json") as f:
            sampleDict = json.loads(f.read())
        f.close()

        for raw in sampleDict:
            raw["c-coefficients"] = np.array(raw["c-coefficients"]).flatten()
            audios_dict.append(raw)
        return audios_dict

    @staticmethod
    def process_data(data_class):
        _data = []
        for sample in data_class:
            _data.append([sample["features"], sample["audio_class"]])
        return _data
