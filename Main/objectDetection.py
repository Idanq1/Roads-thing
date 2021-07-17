from imutils.video import FPS
import imutils
import cv2
import random
import pyautogui
import time

R = random.randint(0, 255)
G = random.randint(0, 255)
B = random.randint(0, 255)

R1 = random.randint(0, 255)
G1 = random.randint(0, 255)
B1 = random.randint(0, 255)

"""
BOOSTING Tracker: Based on the same algorithm used to power the machine learning behind Haar cascades (AdaBoost), 
but like Haar cascades, is over a decade old. This tracker is slow and doesn’t work very well. Interesting only for 
legacy reasons and comparing other algorithms.

MIL Tracker: Better accuracy than BOOSTING tracker but does a poor job of reporting failure.

KCF Tracker: Kernelized Correlation Filters. Faster than BOOSTING and MIL. Similar to MIL and KCF, does not handle full 
occlusion well.
CSRT Tracker: Discriminative Correlation Filter (with Channel and Spatial Reliability). Tends to be more accurate than 
KCF but slightly slower.

MedianFlow Tracker: Does a nice job reporting failures; however, if there is too large of a jump in motion, such as fast 
moving objects, or objects that change quickly in their appearance, the model will fail.

TLD Tracker: I’m not sure if there is a problem with the OpenCV implementation of the TLD tracker or the actual 
algorithm itself, but the TLD tracker was incredibly prone to false-positives. Not recommended.


MOSSE Tracker: Very, very fast. Not as accurate as CSRT or KCF but a good choice if you need pure speed.

GOTURN Tracker: The only deep learning-based object detector included in OpenCV. It requires additional model files to 
run (will not be covered in this post). My initial experiments showed it was a bit of a pain to use even though it 
reportedly handles viewing changes well (my initial experiments didn’t confirm this though). I’ll try to cover it in a 
future post, but in the meantime, take a look at Satya’s writeup.
"""


def in_box(position, rectangle):
    x = position[0]
    y = position[1]

    x1 = rectangle[0]
    x2 = rectangle[2]
    y1 = rectangle[1]
    y2 = rectangle[3]

    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


def get_speed(frame, x, y):
    global got_first
    global got_second
    global start_time
    global ignore

    x1 = (250, 400)  # Black
    y1 = (998, 430)  # Red
    r1 = (x1[0], x1[1], y1[0], y1[1])

    x2 = (500, 300)  # Red
    y2 = (998, 330)  # Black
    r2 = (x2[0], x2[1], y2[0], y2[1])

    # cv2.putText(frame, "Y DOWN", (350, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    # cv2.putText(frame, "Y UP", (300, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    # cv2.putText(frame, "X UP", (850, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    # cv2.putText(frame, "X DOWN", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 3)

    # cv2.rectangle(frame, x1, y1, (255, 255, 255), 3)
    # cv2.rectangle(frame, x2, y2, (255, 255, 255), 3)

    # cv2.circle(frame, x2, 1, (0, 0, 0), 2)  # Black ----------------
    # cv2.circle(frame, y2, 1, (0, 0, 255), 2)  # R   ----------------

    if in_box((x, y), r1):  # First
        got_first = True

    if in_box((x, y), r2) and ignore:
        print(time.time() - start_time)
        got_second = True
        got_first = False

    if got_first:
        got_first = False
        start_time = time.time()
        got_second = False


def main(tracker, vc):
    box = None
    fps = None
    while True:
        frame = vc.read()
        frame = frame[1]
        frame = imutils.resize(frame, width=frame.shape[1]-250)  # resize so it can process faster
        frame = imutils.resize(frame, width=1000)  # resize so it can process faster
        (H, W) = frame.shape[:2]
        print(frame.shape)

        if box is not None:
            (success, box) = tracker.update(frame)  # Grab the new box coordinates

            if success:  # Tracking success
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (R, G, B), 2)
                circle_x = x + int(w/2)
                circle_y = y + int(h/2)
                cv2.circle(frame, (circle_x, circle_y), int(h/8), (0, 0, 0), 2)
                # print(x, y, w, h)

                get_speed(frame, circle_x, circle_y)

            fps.update()
            fps.stop()

            info = [("Tracker", tracker), ("Success", success), ("FPS", "{:.2f}".format(fps.fps()))]

            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (R1, G1, B1), 2)

        cv2.imshow("Video", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('s'):
            box = cv2.selectROI("Video", frame, fromCenter=False, showCrosshair=True)

            tracker.init(frame, box)
            fps = FPS().start()

        elif k == ord('q'):
            break

        elif k == ord('o'):
            print(pyautogui.position())

        elif k == ord('p'):
            while True:
                k = cv2.waitKey(1) & 0xFF
                if k == ord('p'):
                    break

    vc.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tracker_type = "csrt"

    got_first = False
    got_second = False
    ignore = False
    start_time = 0

    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,  # Not recommended
        "mil": cv2.TrackerMIL_create,  # Not recommended
        "tld": cv2.TrackerTLD_create,  # Not recommended
        "medianflow": cv2.TrackerMedianFlow_create,  # Not recommended
        "mosse": cv2.TrackerMOSSE_create  # For fast situations
    }
    video_source = r"../Sources/test2.mp4"
    tracker_type = OPENCV_OBJECT_TRACKERS[tracker_type]()
    video_capture = cv2.VideoCapture(video_source)

    main(tracker_type, video_capture)
