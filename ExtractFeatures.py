from ImportDataset import ImportDataset
from MFCC.AudioTransformMFCC import AudioTransformMFCC

path_to_audios = "D:/Segundo_Semestre/Machine_learning/final_project/archive/audios/"
data_directory = "data/"

#Extracting features for class: arabian
arabian = ImportDataset(path_to_audios, path_to_audios + '_arabic.txt', label=0, transform=AudioTransformMFCC(13))
arabian.__generate_features_files__(data_directory)

#Extracting features for class: english
english = ImportDataset(path_to_audios, path_to_audios + '_english.txt', label=1, transform=AudioTransformMFCC(13))
english.__generate_features_files__(data_directory)

#Extracting features for class: spanish
spanish = ImportDataset(path_to_audios, path_to_audios + '/_spanish.txt', label=2, transform=AudioTransformMFCC(13))
spanish.__generate_features_files__(data_directory)
