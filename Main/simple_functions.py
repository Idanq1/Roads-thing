import cv2
import yolo_image_return as yir
import math


def run_yolo(image, min_confidence=0.2, threshold=0.5, resize=None):
    """
    Returns (image_yolo, image_boxes)
    :param image:
    :param min_confidence:
    :param threshold:
    :param resize:
    :return:
    """
    if resize:
        image = resize_image(image, resize)
    return yir.yolo(image, min_confidence, threshold)


def resize_image(image, resize_prc=50):
    """
    Resizes the image with given amount of resize
    :param image:
    :param resize_prc:
    :return:
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    width = int(image.shape[1] * resize_prc / 100)
    height = int(image.shape[0] * resize_prc / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def calc_area(box):
    """
    Returns the area for a box shaped as (x, y, xToAdd, yToAdd)
    :param box:
    :return:
    """
    b1 = box[0] + box[2]
    b2 = box[1] + box[3]
    return b1 * b2


def calc_area_v2(box):
    """
    Returns the area for box shapes as [(x, y), (xNew, yNew)]
    :param box:
    :return:
    """
    z1 = box[1][0] - box[0][0]
    z2 = box[1][1] - box[0][1]
    return z1*z2


def get_middle_v2(box):
    """
    Returns the middle for boxes shaped as (x, y, xToAdd, yToAdd)
    :param box:
    :return: (x, y)
    :rtype list:
    """
    return int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)


def boxes_to_list(boxes, ignore=None):
    """
    Parse the dictionary to list
    :param ignore:
    :type ignore: list
    :param boxes:
    :return:
    """
    if ignore and not isinstance(ignore, list):
        print("ignore has to be a list")
        ignore = []
    elif not ignore:
        ignore = []
    to_return = []
    for label in boxes:
        if label in ignore:
            continue
        for box in boxes[label]:
            to_return.append(box)

    return to_return


def get_distance(p1, p2):
    """
    Returns the distance between 2 points, distance by pixels
    :param p1: Point 1.
    :type p1: tuple
    :param p2: Point 2
    :type p2: tuple
    :return: int
    """
    dis = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    return dis
