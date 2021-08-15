from ImportDataset import ImportDataset
from AudioTransformMFCC import AudioTransformMFCC


audios_arabian = ImportDataset('D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/',
                        'D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/_arabian.txt',
                        transform=AudioTransformMFCC())
audios_arabian.__generate_features_files__()


audios_dutch = ImportDataset('D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/',
                        'D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/_dutch.txt',
                        transform=AudioTransformMFCC())
audios_dutch.__generate_features_files__()

audios_english_australia = ImportDataset('D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/',
                        'D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/_english_australia.txt',
                        transform=AudioTransformMFCC())
audios_english_australia.__generate_features_files__()

audios_english_usa = ImportDataset('D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/',
                        'D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/_english_usa.txt',
                        transform=AudioTransformMFCC())
audios_english_usa.__generate_features_files__()

audios_french = ImportDataset('D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/',
                        'D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/_french.txt',
                        transform=AudioTransformMFCC())
audios_french.__generate_features_files__()

audios_german = ImportDataset('D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/',
                        'D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/_german.txt',
                        transform=AudioTransformMFCC())
audios_german.__generate_features_files__()

audios_korean = ImportDataset('D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/',
                        'D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/_korean.txt',
                        transform=AudioTransformMFCC())
audios_korean.__generate_features_files__()


audios_russian = ImportDataset('D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/',
                        'D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/_russian.txt',
                        transform=AudioTransformMFCC())
audios_russian.__generate_features_files__()

audios_spanish = ImportDataset('D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/',
                        'D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/_spanish.txt',
                        transform=AudioTransformMFCC())
audios_spanish.__generate_features_files__()

audios_turkish = ImportDataset('D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/',
                        'D:/Segundo_Semestre/Machine_learning/final_project/archive/audios_wav/_turkish.txt',
                        transform=AudioTransformMFCC())
audios_turkish.__generate_features_files__()


