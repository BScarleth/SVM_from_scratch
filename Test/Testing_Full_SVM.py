from Helper import Helper
from Algorithms.FullSVM import SVM
from Algorithms.OneVSOne import OneVsOne

#Reading data features
data_directory = "../data/"
arabic  = Helper.read_features_file(data_directory + "arabic")[:100]
english = Helper.read_features_file(data_directory + "english")[:100]
spanish = Helper.read_features_file(data_directory + "spanish")[:100]

#Preparing data for testing data
fraction = int(len(arabic) * 80 / 100)
data_training = {
    "arabic": Helper.process_data(arabic[fraction:]),
    "english": Helper.process_data(english[fraction:]),
    "spanish": Helper.process_data(spanish[fraction:]),
}

data_testing = Helper.process_data(
    arabic[:fraction]+
    spanish[:fraction]+
    english[:fraction]
)

#Testing the full version of the SMO with a one vs one approach.
one_vs_one = OneVsOne(["arabic", "spanish", "english"], SVM("poly", 2), data_training, data_testing)
one_vs_one.multiclass_classification()