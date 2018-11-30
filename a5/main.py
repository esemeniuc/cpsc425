# Starter code prepared by Borna Ghotbi, Polina Zablotskaia, and Ariel Shann for Computer Vision
# based on a MATLAB code by James Hays and Sam Birch
import time

import numpy as np
from util import sample_images, build_vocabulary, get_bags_of_sifts, generate_confusion_matrix, plotHistogram
from classifiers import nearest_neighbor_classify, svm_classify, knn_score, svm_score

# For this assignment, you will need to report performance for sift features on two different classifiers:
# 1) Bag of sift features and nearest neighbor classifier
# 2) Bag of sift features and linear SVM classifier

# For simplicity you can define a "num_train_per_cat" vairable, limiting the number of
# examples per category. num_train_per_cat = 100 for intance.

# Sample images from the training/testing dataset.
# You can limit number of samples by using the n_sample parameter.

print('Getting paths and labels for all train and test data\n')
# train_image_paths, train_labels, train_available_labels = sample_images("sift/train", n_sample=13)  # FIXME
train_image_paths, train_labels, train_available_labels = sample_images("sift/train", n_sample=500)
test_image_paths, test_labels, test_available_labels = sample_images("sift/test", n_sample=100)

''' Step 1: Represent each image with the appropriate feature
 Each function to construct features should return an N x d matrix, where
 N is the number of paths passed to the function and d is the 
 dimensionality of each image representation. See the starter code for
 each function for more details. '''

# TODO: You code build_vocabulary function in util.py
print('build_vocabulary()')

start = time.time()
# kmeans = build_vocabulary(train_image_paths, vocab_size=20)  # FIXME
kmeans = build_vocabulary(train_image_paths, vocab_size=400)
end = time.time()
print('build_vocabulary time:', end - start)

# TODO: You code get_bags_of_sifts function in util.py
print('get_bags_of_sifts()\n')
start = time.time()
train_image_feats = get_bags_of_sifts(train_image_paths, kmeans)
test_image_feats = get_bags_of_sifts(test_image_paths, kmeans)
end = time.time()
print('get_bags_of_sifts() time:', end - start)

plotHistogram(train_image_feats, train_labels, train_available_labels)

# If you want to avoid recomputing the features while debugging the
# classifiers, you can either 'save' and 'load' the extracted features
# to/from a file.

''' Step 2: Classify each test image by training and using the appropriate classifier
 Each function to classify test features will return an N x l cell array,
 where N is the number of test cases and each entry is a string indicating
 the predicted one-hot vector for each test image. See the starter code for each function
 for more details. '''

print('Using nearest neighbor classifier to predict test set categories\n')
# TODO: YOU CODE nearest_neighbor_classify function from classifers.py
k_param = knn_score(train_image_feats, train_labels)
# k_param = 7
print('K param',k_param)
pred_labels_knn = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k_param)

print('Using support vector machine to predict test set categories\n')
# TODO: YOU CODE svm_classify function from classifers.py
svm_param = svm_score(train_image_feats, train_labels)
# svm_param = 5
print('svm param',svm_param)
pred_labels_svm = svm_classify(train_image_feats, train_labels, test_image_feats, svm_param)

print('---Evaluation---\n')
# Step 3: Build a confusion matrix and score the recognition system for 
#         each of the classifiers.
# TODO: In this step you will be doing evaluation. 
# 1) Calculate the total accuracy of your model by counting number
#   of true positives and true negatives over all. 
# 2) Build a Confusion matrix and visualize it. 
#   You will need to convert the one-hot format labels back
#   to their category name format.
knn_correct = np.sum(pred_labels_knn == test_labels)
svm_correct = np.sum(pred_labels_svm == test_labels)

print("KNN Accurracy =", knn_correct / len(test_labels))
print("SVM Accurracy =", svm_correct / len(test_labels))
knn_cm = generate_confusion_matrix(test_labels, pred_labels_knn, test_available_labels, "KNN")
svn_cm = generate_confusion_matrix(test_labels, pred_labels_svm, test_available_labels, "SVM")

# Interpreting your performance with 100 training examples per category:
#  accuracy  =   0 -> Your code is broken (probably not the classifier's
#                     fault! A classifier would have to be amazing to
#                     perform this badly).
#  accuracy ~= .10 -> Your performance is chance. Something is broken or
#                     you ran the starter code unchanged.
#  accuracy ~= .50 -> Rough performance with bag of SIFT and nearest
#                     neighbor classifier. Can reach .60 with K-NN and
#                     different distance metrics.
#  accuracy ~= .60 -> You've gotten things roughly correct with bag of
#                     SIFT and a linear SVM classifier.
#  accuracy >= .70 -> You've also tuned your parameters well. E.g. number
#                     of clusters, SVM regularization, number of patches
#                     sampled when building vocabulary, size and step for
#                     dense SIFT features.
#  accuracy >= .80 -> You've added in spatial information somehow or you've
#                     added additional, complementary image features. This
#                     represents state of the art in Lazebnik et al 2006.
#  accuracy >= .85 -> You've done extremely well. This is the state of the
#                     art in the 2010 SUN database paper from fusing many 
#                     features. Don't trust this number unless you actually
#                     measure many random splits.
#  accuracy >= .90 -> You used modern deep features trained on much larger
#                     image databases.
#  accuracy >= .96 -> You can beat a human at this task. This isn't a
#                     realistic number. Some accuracy calculation is broken
#                     or your classifier is cheating and seeing the test
#                     labels.
