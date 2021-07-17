import cv2
import yolo_image_return as yir
total_boxes = 0


def run_yolo(images):
    global total_boxes
    m = 0
    images_after_yolo = []

    for image in images:
        img, all_boxes = yir.yolo(image)
        # print(all_boxes)
        cv2.imshow(str(m), image)
        images_after_yolo.append(image)

        car_m = 0
        total_boxes += len(all_boxes)
        print(all_boxes)
        for label in all_boxes.values():
            for car in label:
                # print(car)
                # print(car[0])
                # print(car[0][0])
                # print((car[0][0] + car[1][0]) // 2)
                # print((car[0][1] + car[1][1]) // 2)
                middle = ((car[0][0] + car[1][0]) // 2, (car[0][1] + car[1][1]) // 2)
                cv2.circle(image, middle, 5, 255, -1)
                car_m += 1
                cv2.rectangle(image, (car[0][0], car[0][1]), (car[1][0], car[1][1]), (234, 203, 92), 1)
                text = f"Car {car_m}"
                cv2.putText(image, text, (car[0][0], car[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (234, 203, 92), 2)
        m += 1
    return images_after_yolo


im = cv2.imread(r'..\sources\out257.png')
im2 = cv2.imread(r"..\sources\out260.png")
im = run_yolo([im])
im2 = run_yolo([im2])

cv2.imshow("a", im[0])
cv2.imshow("b", im2[0])
cv2.waitKey(-1)
