from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
import ncc
from functools import reduce

scaleFactor = 0.75


# creates a list of images with each image being 3/4 the size of the previous
def MakePyramid(im: Image, minSIze: (int, int)) -> list:
    resizedImages = []
    minx, miny = minSIze
    x, y = im.size
    while x > minx and y > miny:
        resized = im.resize((x, y), Image.BICUBIC)
        resizedImages.append(resized)
        x = int(x * scaleFactor)
        y = int(y * scaleFactor)

    return resizedImages


# creates a collage of images from a list of images
def ShowPyramid(pyramid: list) -> None:
    assert (len(pyramid) > 0)
    width, height = reduce((lambda acc, elem: (acc[0] + elem.size[0], acc[1] + elem.size[1])), pyramid, (0, 0))
    image = Image.new("L", (width, height), 255)

    image.paste(pyramid[0])
    offset_x = pyramid[0].size[0]
    offset_y = 0
    for im in pyramid[1:]:
        image.paste(im, (offset_x, offset_y))
        offset_y += im.size[1]

    image.show()


# displays the judybats example
def demoShowPyramid() -> None:
    img = Image.open("faces/judybats.jpg")
    ShowPyramid(MakePyramid(img, (100, 100)))


# returns a list of tuples where greater than threshold
def getCenters(correlArr: np.ndarray, threshold: float, unscaleFactor: float) -> list:
    output = []
    h, w = correlArr.shape  # flip w,h values because ncc changes them

    for y in range(h):
        for x in range(w):
            if correlArr[y][x] >= threshold:
                output.append((int(x * unscaleFactor), int(y * unscaleFactor)))

    return output


# draws a box on drawable from topLeft to bottomRight
def drawBox(drawable: ImageDraw, topLeft: tuple, bottomRight: tuple, lineWidth: int) -> None:
    # print("drawing on: ", topLeft, " to ", bottomRight)
    xl, yt = topLeft
    xr, yb = bottomRight
    drawable.line((xl, yt, xr, yt), fill="red", width=lineWidth)  # top
    drawable.line((xl, yt, xl, yb), fill="red", width=lineWidth)  # left
    drawable.line((xl, yb, xr, yb), fill="red", width=lineWidth)  # bottom
    drawable.line((xr, yt, xr, yb), fill="red", width=lineWidth)  # right


# marks faces with red boxes
def FindTemplate(pyramid: list, template: Image, threshold: float) -> Image:
    assert (len(pyramid) > 0)
    templateWidth = 15
    lineWidth = 2
    template.thumbnail((templateWidth, template.size[1]), Image.BICUBIC)
    finalImg = pyramid[0].copy().convert('RGB')
    drawable = ImageDraw.Draw(finalImg)
    width, height = finalImg.size

    for i, img in enumerate(pyramid):
        numpy2dArray = ncc.normxcorr2D(img, template)
        upscaleFactor = (1 / scaleFactor) ** i
        bound = int(upscaleFactor * template.size[0])
        centers = getCenters(numpy2dArray, threshold, upscaleFactor)
        for (x, y) in centers:
            topLeft = (max(x - bound, 0), max(y - bound, 0))
            bottomRight = (min(x + bound, width - 1), min(y + bound, height - 1))
            drawBox(drawable, topLeft, bottomRight, lineWidth)

    return finalImg


# demos the images
def demoFindTemplate() -> None:
    threshold = 0.6
    template = Image.open("faces/template.jpg")

    # family = FindTemplate(MakePyramid(Image.open("faces/family.jpg"), (75, 75)), template, 0.6)
    family = FindTemplate(MakePyramid(Image.open("faces/family.jpg"), (75, 75)), template, threshold)
    family.show()

    # fans = FindTemplate(MakePyramid(Image.open("faces/fans.jpg"), (75, 75)), template, 0.62)
    fans = FindTemplate(MakePyramid(Image.open("faces/fans.jpg"), (75, 75)), template, threshold)
    fans.show()

    # judy = FindTemplate(MakePyramid(Image.open("faces/judybats.jpg"), (75, 75)), template, 0.7)
    judy = FindTemplate(MakePyramid(Image.open("faces/judybats.jpg"), (75, 75)), template, threshold)
    judy.show()

    # sports = FindTemplate(MakePyramid(Image.open("faces/sports.jpg"), (75, 75)), template, 0.52)
    sports = FindTemplate(MakePyramid(Image.open("faces/sports.jpg"), (75, 75)), template, threshold)
    sports.show()

    # students = FindTemplate(MakePyramid(Image.open("faces/students.jpg"), (75, 75)), template, 0.7)
    students = FindTemplate(MakePyramid(Image.open("faces/students.jpg"), (75, 75)), template, threshold)
    students.show()

    # tree = FindTemplate(MakePyramid(Image.open("faces/tree.jpg"), (75, 75)), template, 0.7)
    tree = FindTemplate(MakePyramid(Image.open("faces/tree.jpg"), (75, 75)), template, threshold)
    tree.show()
