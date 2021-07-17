import numpy as np
import time
import cv2
import os


def yolo(source_image, min_confidence=0.2, threshold=0.2, save="", show=True):
    yolo_path = "D:\\Python\\cv2\\yolo-object-detection\\yolo-coco"
    labels_path = f"{yolo_path}\\coco.names"
    labels = open(labels_path).read().strip().split("\n")

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    weights_path = os.path.sep.join([yolo_path, "yolov3.weights"])
    config_path = os.path.sep.join([yolo_path, "yolov3.cfg"])

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    image = cv2.imread(source_image)
    if image is None:
        print("Couldn't find that image")
        return

    cv2.imshow("test", image)
    rectangles = []
    while True:
        box = cv2.selectROI("Test", image)
        print(box)
        box = (box[0], box[1], box[0] + box[2], box[1] + box[3])
        rectangles.append(box)
        print(box)
        # crop_img = image[box[1]: box[3], box[0]: box[2]]
        # cv2.imshow("cropped", crop_img)

        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 55, 255), 2)

        k = cv2.waitKey(0)
        print(k)
        if k == ord('q'):
            break

    for box in rectangles:
        (H, W) = image.shape[:2]
        print(H, W)

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        crop = image[box[1]: box[3], box[0]: box[2]]
        blob = cv2.dnn.blobFromImage(crop, 1 / 255, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layer_outputs = net.forward(ln)
        end = time.time()

    print("YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    classIDs = []
    m = 0

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > min_confidence:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            label = labels[classIDs[i]]
            label_color = (colors[classIDs[i]])

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in label_color]
            if label != "d":
                m += 1
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(label, confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if save != "":
        cv2.imwrite(save, image)
    if show:
        cv2.imshow("Image", image)
    cv2.waitKey(0)


yolo("..\\Sources\\cars2.jpg", threshold=0.2, show=True)