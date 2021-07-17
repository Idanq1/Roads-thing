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


def main(tracker, vc):
    boxes = []
    fps = None
    frame = vc.read()
    while True:
        box = cv2.selectROI("MultiTracker", frame)
        frame = vc.read()
        boxes.append(box)
        multi_tracker = cv2.MultiTracker_create()

        for box in boxes:
            multi_tracker.add(createTrackerByName(tracker), frame, box)

        # frame = frame[1]
        # frame = imutils.resize(frame, width=frame.shape[1]-250)  # resize so it can process faster
        # frame = imutils.resize(frame, width=1000)  # resize so it can process faster
        # (H, W) = frame.shape[:2]
        #
        # if box is not None:
        #     (success, box) = tracker.update(frame)  # Grab the new box coordinates
        #
        #     if success:  # Tracking success
        #         (x, y, w, h) = [int(v) for v in box]
        #         cv2.rectangle(frame, (x, y), (x + w, y + h), (R, G, B), 2)
        #
        #     fps.update()
        #     fps.stop()
        #
        #     info = [("Tracker", tracker), ("Success", success), ("FPS", "{:.2f}".format(fps.fps()))]
        #
        #     for (i, (k, v)) in enumerate(info):
        #         text = "{}: {}".format(k, v)
        #         cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (R1, G1, B1), 2)
        #
        # cv2.imshow("Video", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            box = cv2.selectROI("Video", frame, fromCenter=False, showCrosshair=True)
            print(box)

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
    tracker_type = "kcf"
    video_source = "object_tracking_example.mp4"

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
    tracker_type = OPENCV_OBJECT_TRACKERS[tracker_type]()
    video_capture = cv2.VideoCapture(video_source)
    multiTracker = cv2.MultiTracker_create()

    main(tracker_type, video_capture)
