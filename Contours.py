import cv2


def nothing(x):
    pass


def get_box_size(p1=None, p2=None):
    return abs(p1[0] - p2[0]) * abs(p1[1] - p2[1])


def get_average_box_size(p1=None, p2=None, done=False):
    global rectangle_size
    if done:
        if not rectangle_size:
            return
        avg_size = int(sum(rectangle_size)/len(rectangle_size))
        rectangle_size = []
        return avg_size

    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    d_x = abs(x1 - x2)  # Distance X
    d_y = abs(y1 - y2)  # Distance Y
    rectangle_size.append(d_x * d_y)


rectangle_size = []
rectangle_list = []
image = cv2.imread("cars2.jpg")
cv2.imshow("D", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(image, (3, 5), 0)  # 91 57
edged = cv2.Canny(blur, 200, 416)  # 863 311

cv2.createTrackbar("Threshold1", "D", 0, 1000, nothing)
cv2.createTrackbar("Threshold2", "D", 0, 1000, nothing)
cv2.createTrackbar("Blur1", "D", 0, 50, nothing)
cv2.createTrackbar("Blur2", "D", 0, 50, nothing)

while True:
    image = cv2.imread("cars2.jpg")
    t1 = cv2.getTrackbarPos("Threshold1", "D")
    t2 = cv2.getTrackbarPos("Threshold2", "D")
    b1 = cv2.getTrackbarPos("Blur1", "D")
    b2 = cv2.getTrackbarPos("Blur2", "D")

    if b1 % 2 == 0:
        b1 += 1
    if b2 % 2 == 0:
        b2 += 1
    blur = cv2.GaussianBlur(image, (b1, b2), 0)  # 91 57
    edged = cv2.Canny(blur, t1, t2)

    cv2.imshow('Canny Edges After Contouring', edged)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        get_average_box_size((x, y), (x + w, y + h))
        print(get_box_size((x, y), (x + w, y + h)))
        rectangle_list.append((x, y), (x + w, y + h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

    print("average size:", get_average_box_size(done=True))

    # print("Number of Contours found = " + str(len(contours)))
    cv2.imshow('Contours', image)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cv2.destroyAllWindows()
