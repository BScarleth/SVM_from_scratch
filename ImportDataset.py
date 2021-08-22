from Helper import Helper
import os
import pandas as pd
import librosa

class ImportDataset:

    def __init__(self, root_dir, control_file, label = 0, transform=None):
        self.root_dir = root_dir
        self.audio_files = Helper.read_file(control_file)
        self.transform = transform
        self.label = label
        self.class_name = Helper.get_class_from_file_name(self.audio_files[0])

    def __len__(self):
        return len(self.audio_files)

    def __generate_features_files__(self, directory):
        print("Extracting features...")
        features_file = open(directory + self.class_name + ".json", "w")

        sample_counter = 0
        list_of_samples = []
        for idx in range(self.__len__()):
            percentage = (idx * 100) / self.__len__()
            if (percentage % 10) == 0.0:
                 print("Progress ======= ", percentage)

            current_sample = self.__getitem__(idx)
            if current_sample is not None:
                list_of_samples.append(current_sample)
                sample_counter += 1

        features_file.write(pd.Series(list_of_samples).to_json(orient='values'))
        features_file.close()
        print(sample_counter, " audio features extracted ! :D", self.class_name)

    def __getitem__(self, idx):
        SAMPLE_RATE = 44100
        audio_name = os.path.join(self.root_dir, self.audio_files[idx])
        data, _ = librosa.core.load(audio_name, sr=SAMPLE_RATE)

        sample = {'sample_rate': SAMPLE_RATE, 'audio': data, 'audio_class': self.label}
        duration = len(sample["audio"]) / sample["sample_rate"]

        if duration < 20 or sample["sample_rate"] != SAMPLE_RATE:
            return None

        return self.transform(sample)
