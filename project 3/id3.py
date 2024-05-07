#!/usr/bin/env python3

# Naomi Igbinovia 
# CSCI 4350 -- OLA1 
# Joshua Phillips 
# November 15, 2023

import sys
import math
import numpy as np

# this class initializes a node within the decision tree
class DecisionNode:
    def __init__(self, terminal = False, category = None, value = None, feature = None, left = None, right = None):
# terminal is a boolean that indicates if the given node is a leaf or not 
        self.terminal = terminal

# category stores which category the given node is assigned to if it's
# not a leaf 
        self.category = category

# value stores the split feature of the given node 
        self.value = value

# feature stores the feature index that's used for splitting 
        self.feature = feature
        
# left stores the left child node, right stores the right child node
        self.left = left
        self.right = right


# this function calculates the entropy of the target column
def calculate_entropy(target_column):

# the total number of instances in the target column is calculated 
    total_instances = len(target_column)

# the number of occurences of each unique value is counted in the 
# target column 
    value_counts = {}
    for value in target_column:
        value_counts[value] = value_counts.get(value, 0) + 1

# entropy is initialized to 0
    entropy = 0

# entropy is caluclated using the Shannon entropy formula 
    for count in value_counts.values():

# the probability of each value is calculated 
        probability = (count / total_instances)

# the entropy is updated using said probability 
        entropy -= (probability * math.log2(probability))

# post calculations, the entropy is returned 
    return entropy


# this function calculates the information gain of a given split 
def calculate_info_gain(data, split_value, split_feature, target_feature):

# the data is split above and below the split value and feature 
    above_rows = data[data[:, split_feature] >= split_value][:, target_feature]
    below_rows = data[data[:, split_feature] < split_value][:, target_feature]

# the total number of instances is calculated 
    total_instances = len(above_rows) + len(below_rows)

# the entropy is calculated for the set of above rows and the set of 
# below rows 
    entropy_above_rows = (len(above_rows) / total_instances) * calculate_entropy(above_rows)
    entropy_below_rows = (len(below_rows) / total_instances) * calculate_entropy(below_rows)

# the weighted entropy of above and below rows sets is calculated 
    weighted_entropy = entropy_above_rows + entropy_below_rows

# the total entropy of above and below rows sets is calculated 
    total_entropy = calculate_entropy(data[:, target_feature])

# the infomation gain is calculated and returned 
    info_gain = total_entropy - weighted_entropy
    return info_gain


# this function finds the best split value for the decision tree 
def find_best_split(data, features, target):

    best_split = None
    max_info_gain = 0

# for each feature, 
    for feature in features:

# the information gain and split value is found for the given feature using the 
# find_split_with_max_gain() function, eventually finding the max information gain 
        info_gain, split_value = find_split_with_max_gain(data, feature, target)

# the best split is uodated based on the given feature having a higher infomation
# gain or not 
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_split = (feature, split_value)
            
# the best split is returned 
    return (best_split[1], best_split[0])


# this function finds the split value hat maximizes information gain for 
# the given feature
def find_split_with_max_gain(data, feature, target):

# the data arrary indicies are sorted based on the feature column
    sorted_indices = np.argsort(data[:, feature])

# the sorted indicies are used to sort the data array 
    sorted_data = data[sorted_indices]

# the values are extracted from the feature
    values = sorted_data[:, feature]

# the potential split points are calculated 
    splits = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
    
    max_info_gain = 0
    best_split_value = 0

# the total entropy of the dataset is calculated 
    total_entropy = calculate_entropy(data[:, target])

# for every potential split point, 
    for split_value in splits:

# the information gain is calaculated for each point 
        info_gain = calculate_info_gain(data, split_value, feature, target)

# the max information gain is updated and the best split value is set according to 
# the max information gain 
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_split_value = split_value
            
    return max_info_gain, best_split_value


# this function tests the decision tree on the test dataset and returns the number 
# of correct predictions 
def test_decision_tree(test_data, target, tree):
    
    num_correct = 0

# for every instance in the test dataset,
    for instance in test_data:

# a traversion starts from the root node, and continues until a terminal node is 
# reached 
        current_node = tree
        while not current_node.terminal:
# based on the feature value of the current instance, the traversion goes left or right 
            if instance[current_node.feature] >= current_node.value:
                current_node = current_node.right
            else:
                current_node = current_node.left
# if the predicted category matches the true category of the instance 
        if instance[target] == current_node.category:
# the prediction count increments 
            num_correct += 1

# the total number of predictions is returned 
    return num_correct


# this function builds the decision tree using recurison 
def build_decision_tree(data, features, target):
    data = np.array(data)

# if all instances have the same target value, 
    if len(set(data[:, target])) == 1:
        
# a terminal node is created 
        return DecisionNode(terminal=True, category=data[0, target])

# if there's no features left to split or all instances have the same feature values,
    if len(features) == 0 or (isinstance(data[:, features], list) and all(data[:, features][0] == x for x in data[:, features])) or (isinstance(data[:, features], np.ndarray) and np.all(data[:, features][1:] == data[:, features][0])):

# a terminal node is created 
        categories = data[:, target].tolist()
        most_common_category = max(set(categories), key=categories.count)
        return DecisionNode(terminal=True, category=most_common_category)

# the best split valye for the given dataset is calculated using the find_best_split() 
# function 
    best_split_value, best_split_feature = find_best_split(data, features, target)

# the data is split based on the base split 
    left_data = data[data[:, best_split_feature] < best_split_value]
    right_data = data[data[:, best_split_feature] >= best_split_value]

# the left and right subtrees are built recursively 
    left_node = build_decision_tree(left_data, features, target)
    right_node = build_decision_tree(right_data, features, target)

# a created decision node is returned representing the best split 
    return DecisionNode(value=best_split_value, feature=best_split_feature, left=left_node, right=right_node)


def main():

# the path of the training datafile is read in, opened and the data is read into a list
# of lists
    training_file_path = sys.argv[1]
    with open(training_file_path, 'r') as file:

# the data is converted into a 2D array of float values 
        training_data = [[float(value) for value in line.strip().split()] for line in file]

# the training data is converted into an array 
    training_data = np.array(training_data)

# the features' indicies are extracted 
    features = list(range(len(training_data[0]) - 1))

# the target feature index is extracted 
    target_feature_index = len(training_data[0]) - 1

# the path of the test datafile is read in, opened and the data is read into a list
# of lists
    test_file_path = sys.argv[2]
    with open(test_file_path, 'r') as file:

# the data is converted into a 2D array of float values 
        test_data = [[float(value) for value in line.strip().split()] for line in file]

# the decision tree is built using the training data 
    decision_tree = build_decision_tree(training_data, features, target_feature_index)

# the decision tree is tested on the test data, the correct number of predictions are 
# counted and printed 
    correct_predictions = test_decision_tree(test_data, target_feature_index, decision_tree)
    print(correct_predictions)


if __name__ == '__main__':
    main()

