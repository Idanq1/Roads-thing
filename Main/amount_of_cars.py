import yolo_image_return as yir
import os
import time
import cv2
import json


# Get amount of cars
def get_amount_cars(image):
    """
    Returns the amount of cars in the image
    :param str/cv2 image: Either the path for the image or cv2 object
    :return int: amount of boxes
    """
    # if isinstance()
    img, boxes = yir.yolo(image)
    if "car" not in boxes:
        return 0
    boxes_amount = len(boxes["car"])

    # cv2.imshow("dwa", img)  # FOR TESTING
    cv2.waitKey(-1)

    return boxes_amount


def main():

    data = {}
    for file_path, dirs, files in os.walk(r"..\pictures_examples", topdown=False):
        for file in files:
            if "__MACOSX" not in file_path:  # I have no idea what the _MAXCOSX means
                # print("R: ", file_path, "N: ", file)
                splt = file_path.split("\\")
                if len(splt) < 4:
                    continue
                city = splt[2]
                level = splt[3]  # Sparse/Dense
                r_file = f"{file_path}\\{file}"  # Relative file path
                file_name, file_ext = os.path.splitext(file)  # File extension

                if file_ext != ".png":
                    continue

                print(r_file)
                cars_amount = get_amount_cars(r_file)
                if city not in data:
                    data[city] = {}
                if level not in data[city]:
                    data[city][level] = {}
                data[city][level][file_name] = cars_amount

    with open("..\\results\\results.json", 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    s = time.time()
    main()
    print(time.time() - s)
