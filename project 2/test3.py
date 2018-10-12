from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
#from mlxtend.classifier import StackingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

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
		if(target <= 18 and target >= 12):
			labelSet.append("14-16")
		elif(target <= 28 and target >= 22):
			labelSet.append("24-26")
		elif(target <= 38 and target >= 32):
			labelSet.append("34-36")
		elif(target <= 48 and target >= 42):
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
# test_attributesSet = get_attributeSet(TEST_DATA_FILE_NAME)

# train_rawTargetSet = get_targetSet(TRAIN_RAW_DATA_FILE, RAW_TARGET_INDEX)
y_train = get_targetSet(TRAIN_DATA_FILE_NAME, TOP_TARGET_INDEX)
y_test = get_targetSet(TEST_DATA_FILE_NAME, TOP_TARGET_INDEX)

train_corpus = get_corpus(TRAIN_RAW_DATA_FILE)
test_corpus = get_corpus(TEST_RAW_DATA_FILE)

# hv = HashingVectorizer(stop_words='english', alternate_sign=False, n_features=30000)
# X_train = hv.transform(train_corpus).toarray()
# X_test = hv.transform(test_corpus).toarray()
# print(len(X_train))
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_corpus).toarray()
X_dev = vectorizer.transform(dev_corpus).toarray()


# print(X_train.shape)

# Univariate feature selection
# ch2 = SelectKBest(chi2, k=5000)
# X_train = ch2.fit_transform(X_train, y_train)
# X_test = ch2.transform(X_test)
# print(X_train.shape)


def benchmark(clf):
	# print(clf)
	model = clf.fit(X_train, y_train)
	# y_predict = model.predict(X_test)
	target_names = ['14-16', '24-26', '34-36', '44-46', '?']

	# updated_NB = set_labelTarget(predicted_NB)
	
	predict_proba = model.predict_proba(X_test)

	

	y_predict = []

	for proba in predict_proba:
		max_value = max(proba)
		if max_value > 0.43:
			i = [key for key, value in enumerate(proba) if value == max_value]
			y_predict.append(target_names[i[0]])
		else:
			y_predict.append('?')

	print("Id,Prediction")
	datafile = csv.reader(open(TEST_DATA_FILE_NAME))
	dataset = list(datafile)
	for i, data in enumerate(y_predict):
		print('%s,%s' % (dataset[i][0], data))
	
	
	# report = classification_report(y_test, y_predict, target_names=target_names)
	# print(report)	
	# cm = metrics.confusion_matrix(y_test, y_predict)
	# print(cm)
	# score = accuracy_score(y_test, y_predict)
	# print(score)
	# print("")

	# return report, cm, score


if __name__=='__main__':
	results = []

	# benchmark(MultinomialNB(alpha=0.5))

	# benchmark(SGDClassifier(loss='modified_huber', penalty='elasticnet', epsilon=0.30))

	# benchmark(RandomForestClassifier(n_estimators=100))

	# benchmark(LogisticRegression(penalty='l2', C=5, solver='sag', multi_class='multinomial'))

	# results.append(benchmark(Pipeline([('feature_selection', SelectFromModel(LogisticRegression(penalty="l1", dual=False,
 #                                                  tol=1e-3))), ('classification', LogisticRegression(penalty='l2', C=5, 
 #                                                  solver='sag', multi_class='multinomial'))])))

	# results.append(benchmark(AdaBoostClassifier(learning_rate=2)))

	# clf1 = MultinomialNB()
	# clf2 = LogisticRegression(penalty='l2', C=5, solver='sag', multi_class='multinomial')
	# clf3 = SGDClassifier(loss='modified_huber', penalty='elasticnet', epsilon=0.30)
	# eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

	# clf1 = KNeighborsClassifier(n_neighbors=15)
	# clf2 = RandomForestClassifier(n_estimators=10)
	# clf3 = MultinomialNB()
	# lr = LogisticRegression(penalty='l2', C=5, solver='sag', multi_class='multinomial')
	# sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
	#                           meta_classifier=lr)

	# sclf.fit(X_train, y_train)cd
	# y_predict = sclf.predict(X_test)

	# score = accuracy_score(y_test, y_predict)
	# print(score)

	# results.append(benchmark(eclf1))

