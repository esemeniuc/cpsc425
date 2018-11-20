import numpy as np
import os
import glob
from sklearn.cluster import KMeans

def build_vocabulary(image_paths, vocab_size):
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
    n_each = int(np.ceil(10000 / n_image))  # You can adjust 10000 if more is desired
    
    # Initialize an array of features, which will store the sampled descriptors
    features = np.zeros((n_image * n_each, 128))

    for i, path in enumerate(image_paths):
        # Load SIFT features from path
        descriptors = np.loadtxt(path, delimiter=',',dtype=float)

        # TODO: Randomly sample n_each features from descriptors, and store them in features

    # TODO: pefrom k-means clustering to cluster sampled SIFT features into vocab_size regions.
    # You can use KMeans from sci-kit learn.
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    
    return kmeans
    
def get_bags_of_sifts(image_paths, kmeans):
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
        descriptors = np.loadtxt(path, delimiter=',',dtype=float)

        # TODO: Assign each descriptor to the closest cluster center

        # TODO: Build a histogram normalized by the number of descriptors

    return image_feats

def sample_images(ds_path, n_sample):
    """ Sample images from the training/testing dataset.

    Parameters
    ----------
    ds_path: path to the training/testing dataset.
             e.g., sift/train or sift/test
    n_sample: the number of images you want to sample from the dataset.
              if None, use the entire dataset. 
    
    Returns
    -------
    image_paths: a (n_sample, 1) array that contains the paths to the descriptors. 
    """
    # Grab a list of paths that matches the pathname
    files = glob.glob(os.path.join(ds_path, "*", "*.txt"))
    n_files = len(files)

    if n_sample == None:
        n_sample = n_files

    # Randomly sample from the training/testing dataset
    # Depending on the purpose, we might not need to use the entire dataset
    idx = np.random.choice(n_files, size=n_sample, replace=False)
    image_paths = np.asarray(files)[idx]
 
    # Get class labels
    classes = glob.glob(os.path.join(ds_path, "*"))
    labels = np.zeros(n_sample)

    for i, path in enumerate(image_paths):
        folder, fn = os.path.split(path)
        labels[i] = np.argwhere(np.core.defchararray.equal(classes, folder))[0,0]

    return image_paths, labels

