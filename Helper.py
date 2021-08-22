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
    def get_class_from_file_name(filename):
        return filename.split('/')[0]

    @staticmethod
    def read_features_file(filename):
        print("Reading file...")
        audios_dict = []

        with open(filename + ".json") as f:
            sampleDict = json.loads(f.read())
        f.close()

        for raw in sampleDict:
            raw["features"] = np.array(raw["features"])
            audios_dict.append(raw)
        print("File ready!")
        return audios_dict

    @staticmethod
    def process_data(data_class):
        _data = []
        for sample in data_class:
            _data.append([sample["features"], sample["audio_class"]])
        return _data

