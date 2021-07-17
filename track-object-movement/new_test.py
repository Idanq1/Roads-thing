import cv2
import random


# Create tracker
def set_tracker(tracker):
    tracker = tracker.lower()
    trackers = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,  # Not recommended
        "mil": cv2.TrackerMIL_create,  # Not recommended
        "tld": cv2.TrackerTLD_create,  # Not recommended
        "medianflow": cv2.TrackerMedianFlow_create,  # Not recommended
        "mosse": cv2.TrackerMOSSE_create  # For fast situations
    }
    if tracker not in trackers:
        print("Couldn't find that tracker")
        return False
    return trackers[tracker]()


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
        multi_tracker.add(tracker, frame, box)
    return multi_tracker


# Process video and follow with trackers(?)
def update(multi_tracker, cap):
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        success, boxes = multi_tracker.update(frame)

        for new_box in boxes:
            color = tuple([random.randint(0, 255) for r in range(3)])
            p1 = (int(new_box[0]), int(new_box[1]))
            p2 = (int(new_box[0] + new_box[2]), int(new_box[1] + new_box[3]))
            cv2.rectangle(frame, p1, p2, color, 2, 1)

        cv2.imshow("Wa", frame)


def main():

    video = "object_tracking_example.mp4"
    cap = cv2.VideoCapture(video)
    frame = cap.read()[1]
    boxes = set_squares("d")
    multi_tracker = create_trackers(boxes, "kCf", frame)
    update(multi_tracker, cap)
    print("Peace")
