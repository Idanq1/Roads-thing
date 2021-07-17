import sys
import cv2
import random
from simple_functions import run_yolo

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def create_tracker_by_name(tracker_type):
    # Create a tracker based on tracker name
    if tracker_type == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


# Create tracker
def set_tracker(tracker):
    tracker = tracker.lower()
    trackers = {
        "csrt": cv2.TrackerCSRT_create(),
        "kcf": cv2.TrackerKCF_create(),
        "boosting": cv2.TrackerBoosting_create(),  # Not recommended
        "mil": cv2.TrackerMIL_create(),  # Not recommended
        "tld": cv2.TrackerTLD_create(),  # Not recommended
        "medianflow": cv2.TrackerMedianFlow_create(),  # Not recommended
        "mosse": cv2.TrackerMOSSE_create()  # For fast situations
    }
    if tracker not in trackers:
        print("Couldn't find that tracker")
        return False
    return trackers[tracker]


# Get squares to follow, for now I'll use ROI
def set_squares(cords):
    print(cords)  # WON'T USE FOR NOW
    cords = [(540, 475, 104, 96), (548, 181, 121, 124), (426, 337, 28, 139)]
    return cords


# Add tracker for each square
def create_trackers(boxes, tracker_name, frame):
    multi_tracker = cv2.MultiTracker_create()
    tracker = set_tracker(tracker_name)
    for box in boxes:
        # print("Adding:", box)
        multi_tracker.add(create_tracker_by_name("KCF"), frame, box)

    # b = multi_tracker.getObjects()
    # print("s", b)

    return multi_tracker


# Process video and follow with trackers(?)
def update(multi_tracker, cap):

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # b = multi_tracker.getObjects()  # First good second shit
        # print("s", b)

        success, boxes = multi_tracker.update(frame)
        for new_box in boxes:
            # color = tuple([random.randint(0, 255) for r in range(3)])
            color = (255, 0, 0)
            p1 = (int(new_box[0]), int(new_box[1]))
            p2 = (int(new_box[0] + new_box[2]), int(new_box[1] + new_box[3]))
            cv2.rectangle(frame, p1, p2, color, 2, 1)

        cv2.imshow("Wa", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Esc pressed
            break


def main():
    video = "test.mp4"
    cap = cv2.VideoCapture(video)
    suc, frame = cap.read()
    _, boxes = run_yolo(frame)
    print(boxes)
    cv2.imshow("dada", _)
    # boxes = set_squares("d")
    # print(boxes)
    # multi_tracker = create_trackers(boxes, "kcf", frame)
    # update(multi_tracker, cap)


main()
