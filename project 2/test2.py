from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import metrics
import csv
import numpy as np

TRAIN_DATA_FILE_NAME = "./COMP30027_2018S1_proj2-data/train_top10.csv"
TRAIN_RAW_DATA_FILE = "./COMP30027_2018S1_proj2-data/train_raw.csv"
DEV_DATA_FILE_NAME = "./COMP30027_2018S1_proj2-data/dev_top10.csv"
DEV_RAW_DATA_FILE = "./COMP30027_2018S1_proj2-data/dev_raw.csv"
TEST_DATA_FILE_NAME = "./COMP30027_2018S1_proj2-data/test_top10.csv"
TEST_RAW_DATA_FILE = "./COMP30027_2018S1_proj2-data/test_raw.csv"
RAW_TARGET_INDEX = 2
TOP_TARGET_INDEX = -1

def get_attributeSet(file_name):
	datafile = csv.reader(open(file_name))
	dataset = list(datafile)
	attributesSet = []
	for data in dataset:
		attribute = data[1:-1]
		attribute = list(map(int, attribute))
		attributesSet.append(attribute)
	return attributesSet

def get_targetSet(file_name, index):
	datafile = csv.reader(open(file_name))
	dataset = list(datafile)
	targetSet = []
	for data in dataset:
		if index == RAW_TARGET_INDEX:
			targetSet.append(int(data[index]))
		elif index == TOP_TARGET_INDEX:
			targetSet.append(data[index])
	return targetSet


def set_labelTarget(original):
    labelSet = []
    for target in original:
        if(target < 22.5 and target >= 10):
            labelSet.append("14-16")
        elif(target <= 30 and target >= 22.5):
            labelSet.append("24-26")
        elif(target <= 40 and target >= 31):
            labelSet.append("34-36")
        elif(target <= 50 and target >= 41):
            labelSet.append("44-46")
        else:
            labelSet.append("?")
    return labelSet

def get_corpus(file_name):
	corpus = []
	datafile = csv.reader(open(file_name))
	dataset = list(datafile)
	for data in dataset:
		corpus.append(data[-1])
	return corpus

# X_train = get_attributeSet(TRAIN_DATA_FILE_NAME)
# X_test = get_attributeSet(DEV_DATA_FILE_NAME)
# # test_attributesSet = get_attributeSet(TEST_DATA_FILE_NAME)

# # train_rawTargetSet = get_targetSet(TRAIN_RAW_DATA_FILE, RAW_TARGET_INDEX)
y_train = get_targetSet(TRAIN_DATA_FILE_NAME, TOP_TARGET_INDEX)
y_test = get_targetSet(DEV_DATA_FILE_NAME, TOP_TARGET_INDEX)
train_rawTargetSet = get_targetSet(TRAIN_RAW_DATA_FILE, RAW_TARGET_INDEX)
train_corpus = get_corpus(TRAIN_RAW_DATA_FILE)
test_corpus = get_corpus(DEV_RAW_DATA_FILE)

hv = HashingVectorizer(stop_words='english', alternate_sign=False, n_features=10000)
X_train = hv.transform(train_corpus).toarray()
X_test = hv.transform(test_corpus).toarray()
# print(len(X_train))
# vectorizer = TfidfVectorizer()
# X_train = vectorizer.fit_transform(train_corpus).toarray()
# X_dev = vectorizer.transform(dev_corpus).toarray()

# print(X_train.shape)

# Univariate feature selection
ch2 = SelectKBest(chi2, k=5000)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)
# print(X_train.shape)


def benchmark(clf):
	# print(clf)
	model = clf.fit(X_train, y_train)
	y_predict = model.predict(X_test)
	# # updated_NB = set_labelTarget(predicted_NB)
	# print("Id,Prediction")
	# datafile = csv.reader(open(TEST_DATA_FILE_NAME))
	# dataset = list(datafile)
	# for i, data in enumerate(y_predict):
	# 	print('%s,%s' % (dataset[i][0], data))
	target_names = ['14-16', '24-26', '34-36', '44-46', '?']
	report = classification_report(y_test, y_predict, target_names=target_names)
	print(report)	
	cm = metrics.confusion_matrix(y_test, y_predict)
	print(cm)
	score = accuracy_score(y_test, y_predict)
	print(score)
	print("")

	return report, cm, score


if __name__=='__main__':
    print("linear_model: LinearRegression")
    clf_LR = LinearRegression()
    model_LR = clf_LR.fit(X_train, train_rawTargetSet)
    predicted_LR = model_LR.predict(X_test)
    updatedTarget_LR = set_labelTarget(predicted_LR)
    print(metrics.confusion_matrix(y_test, updatedTarget_LR))
    print(accuracy_score(updatedTarget_LR, y_test))
    print("")
