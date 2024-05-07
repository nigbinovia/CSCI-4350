#!/usr/bin/env python3

# Naomi Igbinovia 
# CSCI 4350 -- OLA1 
# Joshua Phillips 
# December 6, 2023

import sys
import math
import random

# this function reads in data from a given file 
def read_data(filename):

# an empty list is created to store data
    data = []

# with the given gile is opened using read mode,
    with open(filename, 'r') as file:

# for every line of the file, 
        for line in file:

# the given line is split into a list of strings, converted into a float, stored in a 
# list comprehension 
            row = [float(x) for x in line.strip().split()]

# the list of floats is appended to declared data list 
            data.append(row)

# the data list (containing a list of all the data points) is returned 
    return data


# this function calculates and returns the Euclidean distance between two 
# given data points 
def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# this function updates the centroids based on the current clusters 
def update_centroids(clusters):

# an empty list is created to store updated centroids
    new_centroids = []

# for each cluter in a list of clusters,
    for cluster in clusters:

# if the cluster is empty, 
        if len(cluster) == 0:

# the centroid is initalized with a random data point 
            random_point = random.choice([point for sublist in clusters for point in sublist])

# the class label is excluded from the random point, and appended to the new 
# centroid list 
            new_centroids.append(random_point[:-1])  

# if the cluster isn't empty, 
        else:

# the number of dimensions in the cluster's data points is calculated
            num_dimensions = len(cluster[0]) - 1 

# the mean of each dimension of the cluster is calculated with list comprehension
            new_centroid = [sum(point[i] for point in cluster) / len(cluster) for i in range(num_dimensions)]

# a new centroid is appended to the new centroids list 
            new_centroids.append(new_centroid)

# an updated centroids list is returned 
    return new_centroids


# this function classifies validation data based on the centroids and 
# and cluster labels
def classify_validation_data(validation_data, centroids, cluster_labels):

# a variable is created to count all the correct classifications
    correct_classifications = 0 

# for every point in the validation data,
    for point in validation_data:

# the distances from the given point to each centroid is calculated using 
# Euclidean distance
        distances = [euclidean_distance(point, centroid) for centroid in centroids] 

# the closest cluster's index is searched for 
        closest_cluster = distances.index(min(distances))

# if the closest centroid's cluster label matches the point's true label, 
        if cluster_labels[closest_cluster] == int(point[-1]):

# the correct classifications count is incremented
            correct_classifications += 1

# the total number of correct classifications is returned 
    return correct_classifications


# this function performs K-means clustering 
def kmeans(training_file, validation_file, k):

# the training and validation data is read in from their files 
    training_data = read_data(training_file)
    validation_data = read_data(validation_file)

# centroids with the first k pointing from the training data is created 
    initial_centroids = training_data[:k]

# the current centroids are created with the inital centroids
    centroids = initial_centroids.copy()

# a variable to store the centroids from older iterations is created
# and set to None
    prev_centroids = None

# while centroids still change, 
    while centroids != prev_centroids:
        prev_centroids = centroids.copy()

# each point is assigned to the closest centroid to form clusters 
        clusters = [[] for _ in range(len(centroids))]

# for each point in the training data, 
        for point in training_data:

# the distances from the given point to each centroid is calculated using Euclidean
# distance 
            distances = [euclidean_distance(point, centroid) for centroid in centroids]

# the closest centroid's index is searched for 
            closest_centroid = distances.index(min(distances))

# the given point is added to the closest centroid's cluster 
            clusters[closest_centroid].append(point)

# the centroids are updated based on the current clusters 
        centroids = update_centroids(clusters)

# cluster labels are assigned by majority vote 
    cluster_labels = []

# for every cluster, 
    for cluster in clusters:
        class_counts = {}

# the occurrences of each class label in the given cluster is counted
        for point in cluster:
            label = int(point[-1])
            class_counts[label] = class_counts.get(label, 0) + 1

# if the cluster isn't empty, 
        if class_counts:

# the class label with the majority vote is searched for
            most_common_label = max(class_counts, key=class_counts.get)

# the majority vote label is appended to the cluster labels list 
            cluster_labels.append(most_common_label)

# if the cluster is empty, 
        else:

# None is appended to the cluster labels list 
            cluster_labels.append(None)
            
# the validation data is classified and the number of correct classifications is counted 
    correct_classifications = classify_validation_data(validation_data, centroids, cluster_labels)

# the total number of correct classifications is returned 
    return correct_classifications

if __name__ == "__main__":

# if the correct number of command-line arguments isn't provided, an error message
# is printed and the program is exited 
    if len(sys.argv) != 4:
        print("Your input was incorrect. Please enter it as the following: ")
        print("python3 kmeans.py <number_of_clusters> <training_data_file> <validation_data_file>")
        sys.exit(1)

# the command-line arguments are parsed 
    k = int(sys.argv[1])
    training_file = sys.argv[2]
    validation_file = sys.argv[3]

# K-means clustering is peformed and the number of correct classifications is printed 
    correct_classifications = kmeans(training_file, validation_file, k)
    print(correct_classifications) 

