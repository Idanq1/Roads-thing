import math
import cv2
import yolo_image_return as yir


def get_middles(boxes):
    """
    Returns the middle of all the boxes
    :param boxes: dict
    :return return_middles: dict
    """

    return_middles = {}
    for key, values in boxes.items():
        for value in values:
            middle = ((value[0][0] + value[1][0]) // 2, (value[0][1] + value[1][1]) // 2)
            if key not in return_middles:
                return_middles[key] = []
            return_middles[key].append(middle)

    return return_middles


def circle_middle(image, boxes_middles):
    """
    Put a circle on an image on certain points (middles)
    :param image: cv2
    :param boxes_middles: dict
    :return: cv2
    """
    for middles in boxes_middles.values():
        for middle in middles:
            cv2.circle(image, middle, 5, (157, 48, 247), -1)
    return image


def distance(p1, p2):
    """
    Returns the distance between 2 points, distance by pixels
    :param p1: tuple
    :param p2: tuple
    :return: int
    """
    dis = math.sqrt(((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2))
    return dis


def dict_values_to_list(d):
    """
    Transfers dictionary to list
    :param d:
    :return:
    """
    to_return = []
    lst = [i for i in d.values()]
    for items in lst:
        for item in items:
            to_return.append(item)
    return to_return


def match_middles(middles1, middles2):
    """
    Match the middles by the least distance
    :param middles1: list
    :param middles2: list
    :return: list
    """
    matched = []
    for middle1 in middles1:
        min_dist = 9999999
        pt2 = None
        if not middles2:
            return matched
        for middle2 in middles2:
            dist = distance(middle1, middle2)
            if dist < min_dist:
                min_dist = dist
                pt2 = middle2
        middles2.remove(pt2)
        matched.append((middle1, pt2))

    print("MATCHED: ", matched)
    return matched


def average_dist(matched_middles):
    """
    Returns the average dist (for each side?)
    :param list matched_middles:
    :return:
    """
    # DOESN'T REALLY WORK SINCE THE USUAL DISTANCE IS 1 BECAUSE IT'S A FRAME DIFFERENT
    # For now I'll do overall average.
    avg = 0
    for middles in matched_middles:
        dist = distance(middles[0], middles[1])
        if dist > 10:
            continue
        avg += dist
        # print(dist)
    avg /= len(matched_middles)
    print(avg)


def resize_image(image, percent):
    """
    Resize the image with given percent
    :param str/cv2 image:
    :param int percent:
    :return cv2:
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def calc_s(boxes):
    """
    Returns dict of all the שטח for each car
    :param boxes:
    :type boxes: dict
    :return:
    :rtype dict:
    """
    boxes_r = []
    for box in boxes["car"]:
        area = (box[1][1] - box[0][1]) * (box[1][0] - box[0][0])
        boxes_r.append(1000000/(area**2))
    return boxes_r


def main():
    # image1_path = r"..\sources\A\34.png"
    # image2_path = r"..\sources\A\40.png"
    # image1_path = r"..\pictures_examples\Hashalom bridge 2,5611\Dense\Screen Shot 2020-02-26 at 16.40.38.png"
    # image2_path = r"..\pictures_examples\Hashalom bridge 2,5611\Dense\Screen Shot 2020-02-26 at 16.40.39.png"
    # image1_path = r"..\pictures_examples\Hashalom bridge 2,5611\Sparse\Screen Shot 2020-02-26 at 8.39.56.png"
    # image2_path = r"..\pictures_examples\Hashalom bridge 2,5611\Sparse\Screen Shot 2020-02-26 at 8.39.57.png"
    image1_path = r"..\pictures_examples\Hashalom bridge 2,5611\Sparse\screenshot-09-03-2020-15.16.1.png"
    image2_path = r"..\pictures_examples\Hashalom bridge 2,5611\Sparse\screenshot-09-03-2020-15.14.2.png"

    resize_percent = 80
    image1 = resize_image(image1_path, resize_percent)
    image2 = resize_image(image2_path, resize_percent)

    image1_y, image1_b = yir.yolo(image1, 0.2, 0.5)
    image2_y, image2_b = yir.yolo(image2, 0.2, 0.5)

    image1_s = calc_s(image1_b)
    image2_s = calc_s(image2_b)

    image1_middles = get_middles(image1_b)
    image2_middles = get_middles(image2_b)
    og_image1 = resize_image(image1_path, resize_percent)
    # og_image2 = resize_image(image1_path, resize_percent)

    circle_middle(image1_y, image1_middles)
    circle_middle(image2_y, image2_middles)

    m = 0
    for box in image1_s:
        cv2.putText(image1_y, str(box), image1_b["car"][m][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (15, 255, 255), 2)
        m += 1

    m = 0
    for box in image2_s:
        cv2.putText(image2_y, str(box), image2_b["car"][m][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (15, 255, 255), 2)
        m += 1

    cv2.imshow("1", image1_y)
    cv2.imshow("2", image2_y)
    cv2.waitKey(-1)

    image1_middles_lst = dict_values_to_list(image1_middles)
    image2_middles_lst = dict_values_to_list(image2_middles)

    matched_middles = match_middles(image1_middles_lst, image2_middles_lst)
    average_dist(matched_middles)
    for matched_middle in matched_middles:
        dist = distance(matched_middle[0], matched_middle[1])
        dist = round(dist)
        # if dist > 30:  # Make better later
        #     continue
        cv2.line(og_image1, matched_middle[0], matched_middle[1], 255, 1)
        cv2.putText(og_image1, str(dist), matched_middle[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Final result", og_image1)
    cv2.imwrite(r"..\output\middles_30_sparse_1.jpg", og_image1)
    cv2.waitKey(-1)
    #
    print(image2_middles)
    print(image1_middles)
    distance(image1_middles_lst[0], image2_middles_lst[0])


main()
