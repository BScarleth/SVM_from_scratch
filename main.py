from Helper import Helper
from sklearn import svm
import random
import numpy as np
from FullSVM import SVM
from OneVSOne import OneVsOne

print("Reading data files...")
arabic = Helper.read_features_file("data/arabic")
dutch = Helper.read_features_file("data/dutch")
#english_australia = Helper.read_features_file("data/english_australia")
#english_usa = Helper.read_features_file("data/english_usa")
#french = Helper.read_features_file("data/french")
#german = Helper.read_features_file("data/german")
#korean = Helper.read_features_file("data/korean")
#russian = Helper.read_features_file("data/russian")
#spanish = Helper.read_features_file("data/spanish")
#turkish = Helper.read_features_file("data/turkish")
print("Files ready!")

_data = {
    "arabic": Helper.process_data(arabic),
    "dutch": Helper.process_data(dutch),
}
#list_of_clases, classifier, data

one_vs_one = OneVsOne([], SVM(), _data)
one_vs_one.multiclass_classification()



