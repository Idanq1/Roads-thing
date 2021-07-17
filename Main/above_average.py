import yolo_image_return as yir
import os
import time
import cv2


# Get amount of cars
def get_amount_cars(image):
    """
    Returns the amount of cars in the image
    :param image: cv2
    :return: int: amount of boxes
    """
    img, boxes = yir.yolo(image)
    if "car" not in boxes:
        return 0
    boxes_amount = len(boxes["car"])

    cv2.imshow("dwa", img)  # FOR TESTING
    cv2.waitKey(-1)

    print(boxes)
    return boxes_amount


# Get average car amount from a file (I'll use a constant for now)
def get_average(road):
    """
    Return the amount of average cars in this specific road
    :param road: str
    :return: int: amount of cars
    """
    print(road)  # Probably use it later with json after I'll start adding the average of every road
    return 30  # Just for now


# Check if the amount of cars is below or above average
def is_above(average, cars):
    """
    Returns true if there are more cars then average, else returns false.
    :param average: int
    :param cars: int
    :return:
    """
    return cars > average


def main():
    image_path = "..\\Sources\\Beit hanan1.jpg"
    average_amount = get_average("42 Beit Hanan")
    cars_amount = get_amount_cars(cv2.imread(image_path))
    print(cars_amount)
    print(is_above(average_amount, cars_amount))


if __name__ == '__main__':
    main()
