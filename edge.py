import cv2
import numpy as np


def nothing(x):
    pass


img = cv2.imread("D:\\Python\\cv2\\Sources\\car2.jpg")

blur = cv2.GaussianBlur(img, (5, 5), 0)
canny_edge = cv2.Canny(blur, 833, 311)
cv2.imwrite("..\\output\\test22.png", canny_edge)

cv2.imshow("Canny", canny_edge)

cv2.createTrackbar("Threshold1", "Canny", 0, 1000, nothing)
cv2.createTrackbar("Threshold2", "Canny", 0, 1000, nothing)
cv2.createTrackbar("Blur1", "Canny", 0, 50, nothing)
cv2.createTrackbar("Blur2", "Canny", 0, 50, nothing)


while True:
    cv2.imshow("Canny", canny_edge)

    k = cv2.waitKey(1) & 0xFF
    if k != 255:
        print(k)
        break

    threshold1 = cv2.getTrackbarPos("Threshold1", "Canny")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Canny")
    blur1 = cv2.getTrackbarPos("Blur1", "Canny")
    blur2 = cv2.getTrackbarPos("Blur2", "Canny")
    if blur1 % 2 == 0:
        blur1 += 1
    if blur2 % 2 == 0:
        blur2 += 1

    blur = cv2.GaussianBlur(img, (blur1, blur2), 0)  # 91 57
    canny_edge = cv2.Canny(blur, threshold1, threshold2)  # 863 311

blur = cv2.GaussianBlur(img, (5, 3), 0)  # 91 57
canny_edge = cv2.Canny(blur, 49, 27)  # 863 311
# contours, h = cv2.findContours(canny_edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# can2 = canny_edge.copy()
# cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
cv2.imshow("Image", img)
# cv2.imshow("Cont", can2)
cv2.waitKey(27)

cv2.destroyAllWindows()
