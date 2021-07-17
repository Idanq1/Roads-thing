import yolo_image_return as yir
import os
import time
import cv2
# import numpy as np

total_boxes = 0


def divide_image(og_image, parts):
    images_total = []
    w = og_image.shape[0]
    h = og_image.shape[1]

    add_w = w / parts
    add_h = h / parts
    for row in range(0, parts):
        for col in range(0, parts):
            images_total.append(og_image[int(row * add_w): int((row + 1) * add_w), int(col * add_h): int((col + 1) * add_h)])

    return images_total


def run_yolo(images):
    global total_boxes
    m = 0
    images_after_yolo = []

    for image in images:
        img, all_boxes = yir.yolo(image)
        # print(all_boxes)
        cv2.imshow(str(m), img)
        images_after_yolo.append(img)

        car_m = 0
        if "car" in all_boxes:
            total_boxes += len(all_boxes["car"])

        for key, values in all_boxes.items():
            for car in values:
                # middle = ((value[0][0] + value[1][0]) // 2, (value[0][1] + value[1][1]) // 2)
                cv2.rectangle(image, (car[0][0], car[0][1]), (car[1][0], car[1][1]), (234, 203, 92), 1)
            # text = f"Car {car_m}"
            # cv2.putText(image, text, (car[0][0], car[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (234, 203, 92), 1)
        m += 1
    return images_after_yolo


def join_images(images, parts):
    mat = [[0 for x in range(0)] for y in range(len(images)//parts)]
    i = 0
    for row in mat:
        for ma in range(parts):
            row.append(images_after_yolo[i])
            i += 1

    new_image = concat_tile(mat)
    cv2.imshow('test', new_image)
    cv2.imwrite("test2.jpg", new_image)
    cv2.waitKey(-1)

    return new_image


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


if __name__ == '__main__':
    start = time.time()

    # original_image = "..\\Sources\\cars.jpg"
    original_image = r"..\pictures_examples\Hashalom bridge 2,5611\Dense\Screen Shot 2020-02-26 at 16.40.39.png"
    image = cv2.imread(original_image)
    parts = 4
    images = divide_image(image, parts)
    images_after_yolo = run_yolo(images)
    join_images(images_after_yolo, parts)

    print(total_boxes)
    print(time.time() - start)
