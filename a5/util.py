import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from typing import TypeVar, Iterable, Tuple, Union, List


def build_vocabulary(image_paths: np.ndarray, vocab_size: int) -> KMeans:
    """ Sample SIFT descriptors, cluster them using k-means, and return the fitted k-means model.
    NOTE: We don't necessarily need to use the entire training dataset. You can use the function
    sample_images() to sample a subset of images, and pass them into this function.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    vocab_size: the number of clusters desired.
    
    Returns
    -------
    kmeans: the fitted k-means clustering model.
    """
    n_image = len(image_paths)

    # Since want to sample tens of thousands of SIFT descriptors from different images, we
    # calculate the number of SIFT descriptors we need to sample from each image.
    n_each = int(np.ceil(20000 / n_image))  # You can adjust 10000 if more is desired

    # Initialize an array of features, which will store the sampled descriptors
    features = np.zeros((0, 128))

    for i, path in enumerate(image_paths):
        # Load SIFT features from path
        descriptors = np.loadtxt(path, delimiter=',', dtype=float)
        # descriptors is n x 128
        # TODO: Randomly sample n_each features from descriptors, and store them in features
        # randomSetOfDescriptors = np.random.RandomState(seed=i).permutation(descriptors)[:n_each]  # fixme
        randomSelection = np.random.choice(descriptors.shape[0], min(n_each, descriptors.shape[0]), replace=False)
        randomSetOfDescriptors = descriptors[randomSelection, :]
        features = np.vstack((features, randomSetOfDescriptors))

    # TODO: perform k-means clustering to cluster sampled SIFT features into vocab_size regions.
    # You can use KMeans from sci-kit learn.
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    # kmeans = KMeans(n_clusters=vocab_size, random_state=0).fit(features)  # fixme
    kmeans = KMeans(n_clusters=vocab_size, n_jobs=8).fit(features)

    return kmeans


def get_bags_of_sifts(image_paths: np.ndarray, kmeans: KMeans) -> np.ndarray:
    """ Represent each image as bags of SIFT features histogram.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    kmeans: k-means clustering model with vocab_size centroids.

    Returns
    -------
    image_feats: an (n_image, vocab_size) matrix, where each row is a histogram.
    """
    n_image = len(image_paths)
    vocab_size = kmeans.cluster_centers_.shape[0]

    image_feats = np.zeros((n_image, vocab_size))

    for i, path in enumerate(image_paths):
        # Load SIFT descriptors
        descriptors = np.loadtxt(path, delimiter=',', dtype=float)
        # TODO: Assign each descriptor to the closest cluster center
        closest = kmeans.predict(descriptors)
        # closest = pairwise_distances_argmin(descriptors, kmeans.cluster_centers_)
        # TODO: Build a histogram normalized by the number of descriptors
        # np.add.at(image_feats[i], closest, 1)
        np.add.at(image_feats[i], closest, 1 / descriptors.shape[0])  # divide to normalize

    return image_feats


def plotHistogram(train_image_feats: np.ndarray, train_labels: np.ndarray, train_available_labels: list) -> None:
    classHistograms = {}  # dict of category to (normalized histogram, number of histograms in that class)
    for i, train_image_category in enumerate(train_labels):  # plot a histogram for each scene category
        # make 15 different histograms, sum them up based on class
        if classHistograms.get(train_available_labels[train_image_category]) is None:
            classHistograms[train_available_labels[train_image_category]] = [train_image_feats[i], 1]
        else:
            np.add(classHistograms[train_available_labels[train_image_category]][0], train_image_feats[i])
            classHistograms[train_available_labels[train_image_category]][1] += 1

    # plot average histogram for each scene category
    for category, (histogram, count) in classHistograms.items():
        # get avg by averaging histograms for each training image
        print(category, count)
        np.divide(histogram, count)
        plt.clf()
        plt.bar(np.arange(train_image_feats.shape[1]), histogram)  # arguments are passed to np.histogram
        plt.title("%s Histogram" % category)
        # plt.show()
        plt.savefig('histogramOutput/' + category + '.png')


def sample_images(ds_path: str, n_sample: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """ Sample images from the training/testing dataset.

    Parameters
    ----------
    ds_path: path to the training/testing dataset.
             e.g., sift/train or sift/test
    n_sample: the number of images you want to sample from the dataset.
              if None, use the entire dataset. 
    
    Returns
    -------
    A tuple containing:
    image_paths: a (n_sample, 1) array that contains the paths to the descriptors.
    labels: a integer value for the label of the image
    classes: a human readable class name
    """
    # Grab a list of paths that matches the pathname
    files = glob.glob(os.path.join(ds_path, "*", "*.txt"))
    n_files = len(files)

    if n_sample is None:
        n_sample = n_files

    # Randomly sample from the training/testing dataset
    # Depending on the purpose, we might not need to use the entire dataset
    # idx = np.random.choice(n_files, size=n_sample, replace=False) #FIXME
    idx = np.random.RandomState(seed=42).choice(n_files, size=n_sample, replace=False)
    image_paths = np.asarray(files)[idx]

    # Get class labels
    classes = glob.glob(os.path.join(ds_path, "*"))
    labels = np.zeros(n_sample)

    for i, path in enumerate(image_paths):
        folder, fn = os.path.split(path)
        labels[i] = np.argwhere(np.core.defchararray.equal(classes, folder))[0, 0]

    return image_paths, labels.astype(int), [class_name.split('/')[2] for class_name in classes]


def generate_confusion_matrix(y_test: np.ndarray,
                              y_pred: np.ndarray,
                              class_names: List[str],
                              predictor_type: str) -> plt:
    # class_names is 15x1 array of floats (one hot)
    plt.clf()
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    title = predictor_type + ' Confusion matrix, without normalization'
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title=title)
    plt.savefig("histogramOutput/" + title + ".png")

    # Plot normalized confusion matrix
    plt.figure()
    title = predictor_type + ' Normalized confusion matrix'
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title=title)

    # plt.show()
    plt.savefig("histogramOutput/" + title + ".png")


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
