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
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
import csv
import numpy as np

TRAIN_DATA_FILE_NAME = "./COMP30027_2018S1_proj2-data/train_top10.csv"
TRAIN_RAW_DATA_FILE = "./COMP30027_2018S1_proj2-data/train_raw.csv"
DEV_DATA_FILE_NAME = "./COMP30027_2018S1_proj2-data/dev_top10.csv"
DEV_RAW_DATA_FILE = "./COMP30027_2018S1_proj2-data/dev_raw.csv"
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




if __name__=='__main__':
	train_attributesSet = get_attributeSet(TRAIN_DATA_FILE_NAME)
	dev_attributesSet = get_attributeSet(DEV_DATA_FILE_NAME)
	train_rawTargetSet = get_targetSet(TRAIN_RAW_DATA_FILE, RAW_TARGET_INDEX)
	train_targetSet = get_targetSet(TRAIN_DATA_FILE_NAME, TOP_TARGET_INDEX)
	dev_targetSet = get_targetSet(DEV_DATA_FILE_NAME, TOP_TARGET_INDEX)


    # Removing features with low variance
	# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
	# new_train_attributesSet = sel.fit_transform(train_attributesSet)
	# new_dev_attributesSet = sel.transform(dev_attributesSet)
	# print(new_train_attributesSet.shape)

    # Univariate feature selection
	# sel = SelectKBest(chi2, k=20)
	# new_train_attributesSet = sel.fit_transform(train_attributesSet, train_targetSet)
	# new_dev_attributesSet = sel.transform(dev_attributesSet)
	# print(new_train_attributesSet.shape)

    # L1-based feature selection
	# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_attributesSet, train_targetSet)
	# model = SelectFromModel(lsvc, prefit=True)
	# new_train_attributesSet = model.transform(train_attributesSet)
	# new_dev_attributesSet = model.transform(dev_attributesSet)


	# print("Naive_bayes")
	# clf_NB = MultinomialNB().fit(train_attributesSet, train_targetSet)
	# predicted_NB = clf_NB.predict(dev_attributesSet)
	# print(metrics.confusion_matrix(dev_targetSet, predicted_NB))
	# print(accuracy_score(predicted_NB, dev_targetSet))
	# print("")


	# print("linear_model: SVM")
	# clf_SVM = SGDClassifier(loss='log', penalty='l2',alpha=1e-3, fit_intercept=True, random_state=42, max_iter=5, tol=None).fit(train_attributesSet, train_targetSet)
	# predicted_SVM = clf_SVM.predict(dev_attributesSet)
	# print(metrics.confusion_matrix(dev_targetSet, predicted_SVM))
	# print(np.mean(predicted_SVM == dev_targetSet))
	# print("")

	# ERROR: inifinite loop, show nothing
	# print("SVM: SVR")
	# clf_SVR = SVR(C=0.1, epsilon=0.1)
	# model_SVR = clf_SVR.fit(train_attributesSet, train_rawTargetSet)
	# predicted_SVR = model_SVR.predict(dev_attributesSet)
	# updatedTarget_SVR = set_labelTarget(predicted_SVR)
	# print(metrics.confusion_matrix(dev_targetSet, updatedTarget_SVR))
	# print(np.mean(updatedTarget_SVR == dev_targetSet))
	# print("")

    # ERROR: inifinite loop, show nothing
	# print("SVM: SVC")
	# clf_SVC = SVC()
	# model_SVC = clf_SVC.fit(train_attributesSet, train_targetSet)
	# predicted_SVC = model_SVC.predict(dev_attributesSet)
	# #updatedTarget_SVC = set_labelTarget(predicted_SVR)
	# print(metrics.confusion_matrix(dev_targetSet, predicted_SVC))
	# print(np.mean(predicted_SVC == dev_targetSet))
	# print("")

	print("linear_model: LinearRegression")
	clf_LR = LinearRegression()
	model_LR = clf_LR.fit(train_attributesSet, train_rawTargetSet)
	print(clf_LR.score(dev_attributesSet, dev_targetSet))
	predicted_LR = model_LR.predict(dev_attributesSet)
	updatedTarget_LR = set_labelTarget(predicted_LR)
	print(metrics.confusion_matrix(dev_targetSet, updatedTarget_LR))
	print(accuracy_score(updatedTarget_LR, dev_targetSet))
	print("")

	# ERROR: inifinite loop, show nothing
	# print("KNN")
	# for k in range(1,20):
	# 	print("n_neighbors=", k, ", ", "weights='distance'")
	# 	clf_KNN = KNeighborsClassifier(n_neighbors=k, weights='distance')
	# 	model_KNN = clf_KNN.fit(train_attributesSet, train_rawTargetSet)
	# 	predicted_KNN = model_KNN.predict(dev_attributesSet)
	# 	updatedTarget_KNN = set_labelTarget(predicted_KNN)
	# 	print(metrics.confusion_matrix(updatedTarget_KNN, dev_targetSet))
	# 	print(accuracy_score(updatedTarget_KNN, dev_targetSet))
	# 	print("")

	# for k in range(1,20):
	# 	print("n_neighbors=", k, ", ", "weights='uniform'")
	# 	clf_KNN = KNeighborsClassifier(n_neighbors=k, weights='uniform')
	# 	model_KNN = clf_KNN.fit(train_attributesSet, train_rawTargetSet)
	# 	predicted_KNN = model_KNN.predict(dev_attributesSet)
	# 	updatedTarget_KNN = set_labelTarget(predicted_KNN)
	# 	print(metrics.confusion_matrix(updatedTarget_KNN, dev_targetSet))
	# 	print(accuracy_score(updatedTarget_KNN, dev_targetSet))
	# 	print("")
	# # print(train_targetSet)
	# # print(len(predicted))
	# print(metrics.confusion_matrix(dev_targetSet, predicted_SVM))
	# print(np.mean(updatedTarget == dev_targetSet))
