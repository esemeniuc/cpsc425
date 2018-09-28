from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
import ncc
from functools import reduce


def MakePyramid(im: Image, minsize: (int, int)) -> list:
    scaleFactor = 0.75
    resizedImages = []
    minx, miny = minsize
    x, y = im.size
    while x > minx and y > miny:
        resized = im.resize((x, y), Image.BICUBIC)
        resizedImages.append(resized)
        x = int(x * scaleFactor)
        y = int(y * scaleFactor)

    return resizedImages


def ShowPyramid(pyramid: list) -> None:
    width, height = reduce((lambda acc, elem: (acc[0] + elem.size[0], acc[1] + elem.size[1])), pyramid, (0, 0))
    image = Image.new("L", (width, height), 255)

    offset_x = 0
    for im in pyramid:
        image.paste(im, (offset_x, 0))
        offset_x += im.size[0]

    image.show()


def demoShowPyramid() -> None:
    img = Image.open("faces/judybats.jpg")
    ShowPyramid(MakePyramid(img, (100, 100)))


# def FindTemplate(pyramid, template, threshold) -> None:
#     draw = ImageDraw.Draw(im)
#     draw.line((x1, y1, x2, y2), fill="red", width=2)
#     del draw
