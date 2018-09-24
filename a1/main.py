from PIL import Image
import numpy as np
import math
from scipy import signal

#returns true if an integer is odd
def isOdd(n: int) -> bool:
    return n % 2 == 1

#returns a boxfilter of size n*n
def boxfilter(n: int) -> np.ndarray:
    assert isOdd(n)
    valueToFill = 1 / (n * n)
    return np.full((n, n), valueToFill)


def nextOdd(n: float) -> int:
    return int(np.ceil(n) // 2 * 2 + 1)

# returns a 1D Gaussian filter for a given value of sigma
def gauss1d(sigma: float) -> np.ndarray:
    length = nextOdd(6 * sigma)
    distFromCenterList: list = [x for x in range(-(length // 2), (length // 2) + 1)]
    unNormalized = [math.exp(- (x ** 2) / (2 * (sigma ** 2))) for x in distFromCenterList]
    listSum = sum(unNormalized)
    return np.asarray([x / listSum for x in unNormalized])

# returns a 2D Gaussian filter for a given value of sigma
def gauss2d(sigma: float) -> np.ndarray:
    f = gauss1d(sigma)
    f = f[np.newaxis]  # make 2d
    fT = f.transpose()
    return np.asarray(signal.convolve(f, fT))

# part 1-4a
# applies Gaussian convolution to a 2D array for the given value of sigma
def gaussconvolve2d(array: np.ndarray, sigma: float) -> np.ndarray:
    filter = gauss2d(sigma)
    return signal.convolve2d(array, filter, 'same')

# part 1-4b
#shows the dog picture as original, and as blurred using gaussian filter
def demo() -> None:
    im = Image.open('dog.jpg')
    im = im.convert('L')  # make grayscale
    im.show()  # display original
    im_array = np.asarray(im)

    im2_array = gaussconvolve2d(im_array, 3)
    im2 = Image.fromarray(im2_array)
    im2.show() #display blurred dog
    return


# part 2-1
# returns a image that is either a high or low pass of a source image
# if hipass==false, then returns a low pass version
def hilowpassrgb(filename: str, sigma: float, hipass=True) -> np.ndarray:
    im = Image.open(filename)
    im = im.convert('RGB')
    # im.show()  # display
    im_array = np.asarray(im)

    im2_array_r = gaussconvolve2d(im_array[:, :, 0], sigma)
    im2_array_g = gaussconvolve2d(im_array[:, :, 1], sigma)
    im2_array_b = gaussconvolve2d(im_array[:, :, 2], sigma)

    im2_array = np.dstack((im2_array_r, im2_array_g, im2_array_b))
    # print('Before:', filename, ' ', np.mean(im2_array))

    if hipass:
        im2_array = np.subtract(im_array, im2_array)
        # im2_array[:,:] += 128
        # print(im2_array)

    # print('After:', filename, ' ', np.mean(im2_array))

    # im2 = Image.fromarray(np.uint8(im2_array))
    # im2.show()
    return im2_array

#shows the hybridized images for part 2-3
def hybrid_demo() -> None:

    #catdog
    hi = hilowpassrgb('images/0a_cat.bmp', 4, hipass=True)
    lo = hilowpassrgb('images/0b_dog.bmp', 8, hipass=False)
    im = Image.fromarray(np.uint8(np.clip(np.add(hi, lo), 0, 255)))
    im.show()

    #marilynstein
    hi = hilowpassrgb('images/2a_einstein.bmp', 5, hipass=True)
    lo = hilowpassrgb('images/2b_marilyn.bmp', 2, hipass=False)
    im = Image.fromarray(np.uint8(np.clip(np.add(hi, lo), 0, 255)))
    im.show()

    #motobike
    hi = hilowpassrgb('images/1b_motorcycle.bmp', 3, hipass=True)
    lo = hilowpassrgb('images/1a_bicycle.bmp', 9, hipass=False)
    im = Image.fromarray(np.uint8(np.clip(np.add(hi, lo), 0, 255)))
    im.show()
