import csv
import copy
import math
import numpy as np


#    Description: this program is for the assignment 1 of
#    machine learning to implement a supervised NB classifier
#    and an unsupervised NB classifier and evaluate their results
#
#    Date: 26 March 2018
#
#    Author: Ruoyi Wang(683436), Qiulei Zhang(734416)
#
#    Version: 3


# read csv file and convert the file into list type
def preprocess(filename):
    datafile = csv.reader(open(filename))
    dataset = list(datafile)
    return dataset

# separate dataset according to different class of target
def separateDataByClass(dataset):
    separatedSet = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separatedSet):
            separatedSet[vector[-1]] = []
        separatedSet[vector[-1]].append(vector)
    return separatedSet

# hold-out strategy to random separate dataset into
# training data and test data
# splitRatio: the radio of training data to dataset
# isRandom: whether to select by random or in sequence
def hold_out_split(dataset, splitRatio, isRandom):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    if(isRandom == True):
       np.random.shuffle(copy)
    while len(trainSet) < trainSize:
        trainSet.append(copy.pop())
    return [trainSet, copy]

# random divide the dataset into equal k parts
def divide_dataset_k(dataset, k):
    trainSize = int(len(dataset) / k)
    dividedSet = {}
    copy = list(dataset)
    for i in range(k):
        np.random.shuffle(copy)
        if i not in dividedSet:
            dividedSet[i] = []
        while len(dividedSet[i]) < trainSize:
            dividedSet[i].append(copy.pop())
    return dividedSet

# cross-validataion strategy, each time, select one part as test data
# and the other parts as training data, and then loop k times to ensure
# each part as test data one time
def k_cross_validation(dataset, k):
    dividedSet = divide_dataset_k(dataset, k)
    newDataSet = {}
    for i in range(k):
        if i not in newDataSet:
            newDataSet[i] = {}
        if "train" not in newDataSet[i]:
            newDataSet[i]["train"] = []
        if "test" not in newDataSet[i]:
            newDataSet[i]["test"] = []
        # one part is test data
        newDataSet[i]["test"] = dividedSet[i]
        # the other part is training data
        for j in range(k):
            if j != i:
                newDataSet[i]["train"].extend(dividedSet[j])
    return newDataSet

# smoothing add-k method to avoid possibility is zero
# unseen events get a count k,
# all counts are increased to ensure to that monotonicity is maintained
# and make the distribution more smoothing and even
def smoothing_add_k(count_priors, count_posteriors,k):
    for index, targetsCount in count_posteriors.items():
        for targetName, attributesCount in targetsCount.items():
            for attributeName, attributeNum in attributesCount.items():
                for priorsKey, priorsValue in count_priors.items():
                    if attributeName not in count_posteriors[index][priorsKey]:
                        # add unseen events
                        count_posteriors[index][priorsKey][attributeName] = 0
                # all attribute value add k to ensure the monotonicity
                count_posteriors[index][targetName][attributeName] += k
    for index, targetsCount in count_posteriors.items():
        for targetName, attributesCount in targetsCount.items():
            for attributeName, attributeNum in attributesCount.items():
                # add k times of the number of attributes values for attribute X
                attributesCount[attributeName] = attributeNum / (count_priors[targetName] + k * len(attributesCount))
    return count_posteriors

# count and calculate the the number of each class value and priors probability
def calculate_priors_supervised(trainingset):
    separated = separateDataByClass(trainingset)
    count_priors = {key:len(value) for key, value in separated.items()}
    priors_dict = {key:value/float(len(dataset)) for key, value in count_priors.items()}
    return [count_priors, priors_dict]

# calculate the posterior probability
def calculate_posteriors_supervised(trainingSet):
    count_posteriors = {}
    for i, val in enumerate(trainingSet):
        for j in range(len(val)-1):
            if j not in count_posteriors:
                count_posteriors[j] = {}
            if val[len(val)-1] not in count_posteriors[j] :
                count_posteriors[j][val[len(val)-1]] = {}
            if val[j] not in count_posteriors[j][val[len(val)-1]]:
                count_posteriors[j][val[len(val)-1]][val[j]] = 1;
            else:
                count_posteriors[j][val[len(val)-1]][val[j]] += 1;
    return count_posteriors

# train the training dataset by supervised NB
# k: the k value of smoothing value
def train_supervised(trainingset,k):
    count_priors, priors_dict = calculate_priors_supervised(trainingset)
    count_posteriors = calculate_posteriors_supervised(trainingset)
    posteriors_dict = smoothing_add_k(count_priors, count_posteriors, k)
    return [priors_dict, posteriors_dict]

# calculate bayes probability using bayes rule
def calculate_bayes(value, priors, posteriors):
    bayes_probability = {}
    for targetName, priorsValue in priors.items():
        for j in range(len(value)-1):
            if targetName not in bayes_probability:
                bayes_probability[targetName] = priorsValue
            if value[j] not in posteriors[j][targetName]:
                posteriors[j][targetName][value[j]] = 0.00001
            bayes_probability[targetName] *= posteriors[j][targetName][value[j]]
    sum_value = sum(bayes_probability.values())
    for key,bayes in bayes_probability.items():
        bayes_probability[key] = bayes / sum_value
    return bayes_probability

# predict the class by supervised NB based on the class with maximum bayes probability
def predict_supervised(testdata, priors, posteriors):
    predictSet = {}
    for i, value in enumerate(testdata):
        bayes_probability = calculate_bayes(value, priors, posteriors)
        max_value = max(bayes_probability.values())
        predictSet[i] = [key for key, value in bayes_probability.items() if value == max_value]
    return predictSet

# evaluate the correct number of predicting class and its accuracy by supervised NB
def evaluate_supervised(predict_data, test):
    num_correct = 0
    for key,value in enumerate(test):
        if (predict_data[key][0] == value[len(value)-1]):
            num_correct = num_correct + 1
        else:
            print(key)
            print(value)
            print(predict_data[key][0])
    accuracy = num_correct / len(test) * 100.0
    print([num_correct, len(test), accuracy])
    return [num_correct, len(test), accuracy]

# label every class of each instances with random probability
# all classes' probability of each instance add to one
def generate_random_labels(testdata):
    random_labels = {}
    targetName = []
    for i, value in enumerate(testdata):
        random_labels[i] = {"attributes":value}
        if value[len(value) - 1] not in targetName:
            targetName.append(value[len(value) - 1])
    for i, dict_value in random_labels.items():
        # create random probability and sum to one
        possibility_array = np.random.dirichlet(np.ones(len(targetName)), size = 1)
        for i, target in enumerate(targetName):
            dict_value[target] = possibility_array[0][i]
    return random_labels

# calculate priors probability
# by adding the corresponding (fractional) value for the class
def calculate_priors_unsupervised(random_dict):
    priors_dict ={}
    count_priors = {}
    for i, dict_value in random_dict.items():
        for attributesTargetName, attributesTargetValue in dict_value.items():
            if attributesTargetName != "attributes":
                if attributesTargetName not in priors_dict:
                    priors_dict[attributesTargetName] = 0
                # adding the corresponding (fractional) value for the class
                priors_dict[attributesTargetName] += attributesTargetValue
    for targetName, possibilityValue in priors_dict.items():
        count_priors[targetName] = possibilityValue
        priors_dict[targetName] = possibilityValue / len(random_dict)
    return [count_priors,priors_dict]

# calculate posterior probability of unsupervised NB
# return the number of attributes value for each class and the posteriors probability
def calculate_posteriors_unsupervised(random_dict, priors_dict):
    posteriors_dict = {}
    count_posteriors = {}
    count = 0
    for i, dict_value in random_dict.items():
        attributes = dict_value["attributes"]
        for j in range(len(attributes)-1):
            if attributes[j] not in posteriors_dict:
                posteriors_dict[j] = {}
                for t, p in priors_dict.items():
                    posteriors_dict[j][t] = {}

    for i, dict_value in random_dict.items():
        attributes = dict_value["attributes"]
        for j in range(len(attributes)-1):
            for targetName, attributes_dict in posteriors_dict[j].items():
                if attributes[j] not in attributes_dict:
                    posteriors_dict[j][targetName][attributes[j]] = dict_value[targetName]
                else:
                    posteriors_dict[j][targetName][attributes[j]] += dict_value[targetName]

    count_posteriors = copy.deepcopy(posteriors_dict)
    for i, value in posteriors_dict.items():
        for t, attributes in value.items():
            for attributesName, attributesValue in attributes.items():
                count_posteriors[i][t][attributesName] /= priors_dict[t]
                posteriors_dict[i][t][attributesName] /= (priors_dict[t]*len(random_dict))
    return [count_posteriors,posteriors_dict]

# calculate the probability of each class for each instance
# k: the value of smoothing add-k
def train_unsupervised_iteration(random_dict, k):
    count_priors, priors_dict = calculate_priors_unsupervised(random_dict)
    count_posteriors, posteriors_dict = calculate_posteriors_unsupervised(random_dict, priors_dict)
    if k != 0:
        posteriors_dict = smoothing_add_k(count_priors, count_posteriors, k)
    return [priors_dict, posteriors_dict]

# iterate to improve the estimate each time until the result is stable
# k: the value of smoothing add-k
def train_unsupervised(random_dict, k):
    priors, posteriors = train_unsupervised_iteration(random_dict, k)
    condition = 1
    while(condition > 0):
        for i, dict_value in random_dict.items():
            # calculate the bayes probability of each instances
            bayes_possibility = calculate_bayes(dict_value["attributes"], priors, posteriors)

            for key, bayes in bayes_possibility.items():
                # taking the new class probability distribution
                random_dict[i][key] = bayes
        # re-estimating the priors and posteriors probability
        priors, posteriors = train_unsupervised_iteration(random_dict, k)
        condition -= 1
    return [priors, posteriors]

# predict the class of each instance by selecting the class
# with maximum bayes probability using unsupervised NB
def predict_unsupervised(random_dict, priors, posteriors):
    predictSet = {}
    for i, dict_value in random_dict.items():
        bayes_possibility = calculate_bayes(dict_value["attributes"], priors, posteriors)
        max_value = max(bayes_possibility.values())
        predictSet[i] = [key for key, value in bayes_possibility.items() if value == max_value]
    return predictSet

# evaluate the correct number of predicting class and its accuracy by unsupervised NB
def evaluate_unsupervised(predict_data, actual_data):
    confusion_matrix = {}
    #print(predict_data)
    #calculate the confusion matrix of unsupervised NB
    for key,value in enumerate(actual_data):
        if predict_data[key][0] not in confusion_matrix:
            confusion_matrix[predict_data[key][0]] = {}
        if value[len(value) - 1] not in confusion_matrix[predict_data[key][0]]:
            confusion_matrix[predict_data[key][0]][value[len(value) - 1]] = 1
        else:
            confusion_matrix[predict_data[key][0]][value[len(value) - 1]] += 1

    print(confusion_matrix)
    #count the maximum number of each column of confusion matric
    num_correct = 0
    for predict_attribute, actual_attributes in confusion_matrix.items():
        num_correct += max(actual_attributes.values())
    # accuracy is the number of correct divided by length of dataset
    accuracy = num_correct / len(actual_data) * 100
    print(num_correct)
    print(accuracy)
    return [num_correct, len(actual_data), accuracy]

# read file
filename = 'breast-cancer.csv'
dataset = preprocess(filename)

#random_dict = generate_random_labels(dataset)
#print(random_dict)
#priors, posteriors = train_unsupervised(random_dict,k = 0)
#print(priors)
#print(posteriors)
#predictSet = predict_unsupervised(random_dict, priors, posteriors)
#print(predictSet)
#evaluate_unsupervised(predictSet, dataset)

# question one: compare supervised NB and unsupervised NB

#splitRatio = 1
with open('breast-cancer-q1.csv', 'w') as file:
    write = csv.writer(file)
    # supervised NB by laplace smoothing
    file.write("supervised NB by laplace smoothing")
    file.write('\n')
    for i in range(1):
        #train, test = hold_out_split(dataset, splitRatio, isRandom = False)
        priors, posteriors = train_supervised(dataset, k = 1)
        predict_data = predict_supervised(dataset, priors, posteriors)
        write.writerow(evaluate_supervised(predict_data, dataset))
    file.write('\n')

    # unsupervised NB by laplace smoothing
    file.write("unsupervised NB by laplace smoothing")
    file.write('\n')
    for i in range(1):
        random_dict = generate_random_labels(dataset)
        priors, posteriors = train_unsupervised(random_dict,k = 1)
        predictSet = predict_unsupervised(random_dict, priors, posteriors)
        write.writerow(evaluate_unsupervised(predictSet, dataset))
    file.write('\n')

# question two: compare different file's supervised NB

##splitRatio = 0.67
#with open('mushroom-q2.csv', 'w') as file:
#    write = csv.writer(file)
#    # supervised NB by laplace smoothing
#    file.write("supervised NB by laplace smoothing")
#    file.write('\n')
##   train, test = hold_out_split(dataset, splitRatio, isRandom = False)
#    priors, posteriors = train_supervised(dataset, k = 1)
#    predict_data = predict_supervised(dataset, priors, posteriors)
#    write.writerow(evaluate_supervised(predict_data, dataset))
#    file.write('\n')

#
## question three
#
#with open('hypothyroid-q3.csv', 'w') as file:
#    write = csv.writer(file)
#    # supervised NB hold-out
#    file.write("supervised NB hold-out")
#    file.write('\n')
#    splitRatio = 0.9
#    for num in range(100):
#        # isRandom: whether select the training data randomly
#        # k is the value of smoothing add-k when k=1 is laplace
#        train, test = hold_out_split(dataset, splitRatio, isRandom = True)
#        priors, posteriors = train_supervised(train, k = 1)
#        predict_data = predict_supervised(test, priors, posteriors)
#        write.writerow(evaluate_supervised(predict_data, test))
#    file.write('\n')
#
#    # supervised NB k-cross-validation
#    file.write("supervised NB k-cross-validation")
#    file.write('\n')
#    k = 10
#    for num in range(100):
#        dataset_k = k_cross_validation(dataset, k)
#        num_correct = 0
#
#        for i in range(k):
#            # 1 is the value of smoothing add-k when 1 is laplace
#            priors, posteriors = train_supervised(dataset_k[i]["train"], k = 1)
#            predict_data = predict_supervised(dataset_k[i]["test"], priors, posteriors)
#            num_correct += evaluate_supervised(predict_data, dataset_k[i]["test"])[0]
#        accuracy = num_correct / (len(dataset_k[i]["test"])*k) * 100
#        write.writerow([accuracy])
#    file.write('\n')
#
#    file.write("supervised NB testing on the training data")
#    file.write('\n')
#    priors, posteriors = train_supervised(dataset, k = 1)
#    predict_data = predict_supervised(dataset, priors, posteriors)
#    write.writerow(evaluate_supervised(predict_data, dataset))

## question four
#
#splitRatio = 0.67
#with open('breast-cancer-q4.csv', 'w') as file:
#    write = csv.writer(file)
#
#    # supervised NB without smoothing
#    file.write("supervised NB without smoothing")
#    file.write('\n')
#    for i in range(10):
#        train, test = hold_out_split(dataset, splitRatio, isRandom = True)
#        priors, posteriors = train_supervised(train, k = 0)
#        predict_data = predict_supervised(test, priors, posteriors)
#        write.writerow(evaluate_supervised(predict_data, test))
#    file.write('\n')
#
#    # supervised NB by add-k(k = 0.5) smoothing
#    file.write("supervised NB by add-k(k = 0.5) smoothing")
#    file.write('\n')
#    for i in range(10):
#        train, test = hold_out_split(dataset, splitRatio, isRandom = True)
#        priors, posteriors = train_supervised(train, k = 0.5)
#        predict_data = predict_supervised(test, priors, posteriors)
#        write.writerow(evaluate_supervised(predict_data, test))
#    file.write('\n')
#
#    # supervised NB by laplace smoothing
#    file.write("supervised NB by laplace smoothing")
#    file.write('\n')
#    for i in range(10):
#        train, test = hold_out_split(dataset, splitRatio, isRandom = True)
#        priors, posteriors = train_supervised(train, k = 1)
#        predict_data = predict_supervised(test, priors, posteriors)
#        write.writerow(evaluate_supervised(predict_data, test))
#    file.write('\n')
#
#    # unsupervised NB without smoothing
#    file.write("unsupervised NB without smoothing")
#    file.write('\n')
#    for i in range(10):
#        random_dict = generate_random_labels(dataset)
#        priors, posteriors = train_unsupervised(random_dict,k = 0)
#        predictSet = predict_unsupervised(random_dict, priors, posteriors)
#        write.writerow(evaluate_unsupervised(predictSet, dataset))
#    file.write('\n')
#
#    # unsupervised NB by add-k(k = 0.5) smoothing
#    file.write("unsupervised NB by add-k(k = 0.5) smoothing")
#    file.write('\n')
#    for i in range(10):
#        random_dict = generate_random_labels(dataset)
#        priors, posteriors = train_unsupervised(random_dict,k = 0.5)
#        predictSet = predict_unsupervised(random_dict, priors, posteriors)
#        write.writerow(evaluate_unsupervised(predictSet, dataset))
#    file.write('\n')
#
#    # unsupervised NB by laplace smoothing
#    file.write("unsupervised NB by laplace smoothing")
#    file.write('\n')
#    for i in range(10):
#        random_dict = generate_random_labels(dataset)
#        priors, posteriors = train_unsupervised(random_dict,k = 1)
#        predictSet = predict_unsupervised(random_dict, priors, posteriors)
#        write.writerow(evaluate_unsupervised(predictSet, dataset))
#    file.write('\n')



