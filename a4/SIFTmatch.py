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
    # im3.show()
    return im3


def isInOrientThreshold(theta1: float, theta2: float, threshold: float) -> bool:
    dTheta = ((theta2 - theta1 + (3 * math.pi)) % (2 * math.pi)) - math.pi
    EPSILON = 1E-6
    return dTheta >= -(threshold + EPSILON) and dTheta <= (threshold + EPSILON)


# keypoint (row, column, scale, orientation).  The orientation
# is in the range [-PI, PI1] radians.
def isConsistent(match1: list, match2: list, thresOrient: float, thresScale: float) -> bool:
    dOrient1 = (match1[0][3] - match1[1][3]) % (2 * math.pi)
    dScale1 = match1[0][2] - match1[1][2]

    dOrient2 = (match2[0][3] - match2[1][3]) % (2 * math.pi)
    dScale2 = match2[0][2] - match2[1][2]

    return isInOrientThreshold(dOrient1, dOrient2, thresOrient) and (
            thresScale * dScale2 <= dScale1 <= (1 / thresScale) * dScale2)


def match(image1: Image, image2: Image, siftThreshold: float, useRansac: bool = True,
          ransacThresOrient: float = math.pi / 6, ransacThresScale: float = 0.5) -> Image:
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

    mat = np.arccos(np.dot(descriptors1, np.transpose(descriptors2)))
    for img1Idx, row in enumerate(mat):
        sortedRowIndexes = np.argsort(row)
        denom = max(row[sortedRowIndexes[1]], 1E-6)  # avoid division by 0
        if (row[sortedRowIndexes[0]] / denom) < siftThreshold:
            matched_pairs.append([keypoints1[img1Idx], keypoints2[sortedRowIndexes[0]]])

    if useRansac is False:
        return DisplayMatches(im1, im2, matched_pairs)

    # ransac
    ransacLargestConsistent = [[]] * 10  # make list of 10 empty lists
    for i in range(10):
        randIndex = random.randrange(len(matched_pairs))
        for elem in matched_pairs:
            if isConsistent(matched_pairs[randIndex], elem, ransacThresOrient, ransacThresScale):
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


def test():
    assert (isInOrientThreshold(-0.75 * math.pi, 0.25 * math.pi, math.pi))
    assert (not isInOrientThreshold(-1 * math.pi, 0.25 * math.pi, math.pi / 2))
    assert (isInOrientThreshold(-0.5 * math.pi, 0.25 * math.pi, math.pi))
    assert (isInOrientThreshold(-1 * math.pi, math.pi, math.pi / 8))  # equal
    assert (isInOrientThreshold(-1 / 6 * math.pi, 1 / 6 * math.pi, math.pi / 3))
    assert (not isInOrientThreshold(-1 / 6 * math.pi, 1 / 6 * math.pi, math.pi / 4))
    assert (isInOrientThreshold(11 / 6 * math.pi, -11 / 6 * math.pi, math.pi / 3))
    assert (isInOrientThreshold(-11 / 6 * math.pi, 1 / 6 * math.pi, math.pi / 3))
    assert (not isInOrientThreshold(11 / 6 * math.pi, -5 / 6 * math.pi, math.pi / 3))
    assert (not isInOrientThreshold(11 / 6 * math.pi, -5 / 3 * math.pi, math.pi / 3))
    assert (isInOrientThreshold(11 / 6 * math.pi, -5 / 3 * math.pi, math.pi))


# Test run...
test()
siftThresholds = [0.40, 0.60, 0.70, 0.75, 0.78, 0.79, 0.80]
for siftThreshold in siftThresholds:
    match('scene', 'book', siftThreshold=siftThreshold, useRansac=False).save(
        'results/sb_' + ("%0.2f" % siftThreshold) + '_out.png')


siftThresholds = [0.78, 0.79, 0.8]
ransacOrientThresholds = [math.pi / 4, math.pi / 5, math.pi / 6, math.pi / 7, math.pi / 8]
ransacScaleThresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
for siftThreshold in siftThresholds:
    for ransacOrientThreshold in ransacOrientThresholds:
        for ransacScaleThreshold in ransacScaleThresholds:
            match('library', 'library2', siftThreshold=siftThreshold, useRansac=True,
                  ransacThresOrient=ransacOrientThreshold,
                  ransacThresScale=ransacScaleThreshold).save(
                'results/ll_sift-' + ("%0.2f" % siftThreshold) +
                '_orient-' + ("%0.2f" % ransacOrientThreshold) + '_scale-' +
                ("%0.2f" % ransacScaleThreshold) + '_out.png')

match('library', 'library2', siftThreshold=0.8, useRansac=True,
                  ransacThresOrient=0.4,
                  ransacThresScale=0.4).save('bestWithRansac.png')
