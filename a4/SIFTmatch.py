import random

from PIL import Image, ImageDraw
import numpy as np
import csv
import math


def ReadKeys(image: Image) -> list:
    """Input an image and its associated SIFT keypoints.

    The argument image is the image file name (without an extension).
    The image is read from the PGM format file image.pgm and the
    keypoints are read from the file image.key.

    ReadKeys returns the following 3 arguments:

    image: the image (in PIL 'RGB' format)

    keypoints: K-by-4 array, in which each row has the 4 values specifying
    a keypoint (row, column, scale, orientation).  The orientation
    is in the range [-PI, PI] radians.

    descriptors: a K-by-128 array, where each row gives a descriptor
    for one of the K keypoints.  The descriptor is a 1D array of 128
    values with unit length.
    """
    im = Image.open(image + '.pgm').convert('RGB')
    keypoints = []
    descriptors = []
    first = True
    with open(image + '.key', 'r') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC, skipinitialspace=True)
        descriptor = []
        for row in reader:
            if len(row) == 2:
                assert first, "Invalid keypoint file header."
                assert row[1] == 128, "Invalid keypoint descriptor length in header (should be 128)."
                count = row[0]
                first = False
            if len(row) == 4:
                keypoints.append(np.array(row))
            if len(row) == 20:
                descriptor += row
            if len(row) == 8:
                descriptor += row
                assert len(descriptor) == 128, "Keypoint descriptor length invalid (should be 128)."
                # normalize the key to unit length
                descriptor = np.array(descriptor)
                descriptor = descriptor / math.sqrt(np.sum(np.power(descriptor, 2)))
                descriptors.append(descriptor)
                descriptor = []
    assert len(keypoints) == count, "Incorrect total number of keypoints read."
    print("Number of keypoints read:", int(count))
    return [im, keypoints, descriptors]


def AppendImages(im1, im2):
    """Create a new image that appends two images side-by-side.

    The arguments, im1 and im2, are PIL images of type RGB
    """
    im1cols, im1rows = im1.size
    im2cols, im2rows = im2.size
    im3 = Image.new('RGB', (im1cols + im2cols, max(im1rows, im2rows)))
    im3.paste(im1, (0, 0))
    im3.paste(im2, (im1cols, 0))
    return im3


def DisplayMatches(im1, im2, matched_pairs):
    """Display matches on a new image with the two input images placed side by side.

    Arguments:
     im1           1st image (in PIL 'RGB' format)
     im2           2nd image (in PIL 'RGB' format)
     matched_pairs list of matching keypoints, im1 to im2

    Displays and returns a newly created image (in PIL 'RGB' format)
    """
    im3 = AppendImages(im1, im2)
    offset = im1.size[0]
    draw = ImageDraw.Draw(im3)
    for match in matched_pairs:
        draw.line((match[0][1], match[0][0], offset + match[1][1], match[1][0]), fill="red", width=2)
    im3.show()
    return im3


# keypoint (row, column, scale, orientation).  The orientation
# is in the range [-PI, PI1] radians.
def isConsistent(match1: list, match2: list) -> bool:
    thresOrient = math.pi / 6
    thresScale = 0.5

    dOrient1 = match1[0][3] - match1[1][3]
    dScale1 = match1[0][2] - match1[1][2]

    dOrient2 = match2[0][3] - match2[1][3]
    dScale2 = match2[0][2] - match2[1][2]

    return (dOrient2 - thresOrient <= dOrient1 <= dOrient2 + thresOrient) and (
            thresScale * dScale2 <= dScale1 <= (1 / thresScale) * dScale2)


def match(image1: Image, image2: Image) -> Image:
    """Input two images and their associated SIFT keypoints.
    Display lines connecting the first 5 keypoints from each image.
    Note: These 5 are not correct matches, just randomly chosen points.

    The arguments image1 and image2 are file names without file extensions.

    Returns the number of matches displayed.

    Example: match('scene','book')
    """
    im1, keypoints1, descriptors1 = ReadKeys(image1)
    im2, keypoints2, descriptors2 = ReadKeys(image2)

    matched_pairs = []

    # descriptors: a K-by-128 array, where each row gives a descriptor
    # for one of the K keypoints.  The descriptor is a 1D array of 128
    # values with unit length.

    if len(descriptors2) < 1:
        print("NEED DESCRIPTORS")
        return

    threshold = 0.7

    mat = np.arccos(np.dot(descriptors1, np.transpose(descriptors2)))
    for img1Idx, row in enumerate(mat):
        sortedRowIndexes = np.argsort(row)
        denom = max(row[sortedRowIndexes[1]], 1E-6)  # avoid division by 0
        if (row[sortedRowIndexes[0]] / denom) < threshold:
            matched_pairs.append([keypoints1[img1Idx], keypoints2[sortedRowIndexes[0]]])

    # ransac
    ransacLargestConsistent = [[]] * 10  # make list of 10 empty lists
    for i in range(10):
        randIndex = random.randrange(len(matched_pairs))
        for elem in matched_pairs:
            if isConsistent(matched_pairs[randIndex], elem):
                ransacLargestConsistent[i].append(elem)

    # find largest
    largestIndex = 0
    largestSize = 0
    for i in range(10):
        currentLength = len(ransacLargestConsistent[i])
        if currentLength > largestSize:
            largestSize = currentLength
            largestIndex = i

    im3 = DisplayMatches(im1, im2, ransacLargestConsistent[largestIndex])
    return im3


# Test run...
match('library', 'library2')
