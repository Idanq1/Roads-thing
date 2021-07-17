import cv2
from simple_functions import run_yolo, boxes_to_list, get_middle_v2, get_distance, calc_area
import numpy as np
import time


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


# Add tracker for each square
def create_trackers(boxes, tracker_name, frame):
    multi_tracker = cv2.MultiTracker_create()
    for box in boxes:
        multi_tracker.add(create_tracker_by_name(tracker_name), frame, box)

    return multi_tracker


def update(multi_tracker, cap):
    """
    Process video and follow with trackers(?)
    :param multi_tracker:
    :param cap:
    :return:
    """

    boxes_log = {}
    m = 0
    re = False
    s = time.time()
    frame_counter = 0
    end_frame = None
    fps = 0

    while cap.isOpened():
        success, frame = cap.read()

        f_start = time.time()

        frame_counter += 1  # FPS calculator
        if time.time() - s > 1:
            fps = frame_counter
            # print(fps)
            s = time.time()
            frame_counter = 0

        if not success:
            break

        success, boxes = multi_tracker.update(frame)
        if re:
            break

        # for new_box, old_box in zip(boxes, boxes_log[m-1]):
        car = 0
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(boxes), 3))
        for new_box in boxes:
            # old_middle = get_middle_v2(old_box)d
            middle = get_middle_v2(new_box)

            p1 = (int(new_box[0]), int(new_box[1]))
            p2 = (int(new_box[0] + new_box[2]), int(new_box[1] + new_box[3]))
            color = [int(num) for num in tuple(colors[car])]
            cv2.rectangle(frame, p1, p2, color, 2, 1)

            cv2.circle(frame, middle, 2, (0, 255, 0), 2)
            car += 1

        # Placing the lines
        frame_n = 0
        for frame_num in boxes_log:
            car = 0
            unmoving_cars = 0
            cars = len(boxes_log[frame_num])
            for old_box in boxes_log[frame_num]:
                middle = get_middle_v2(old_box)
                if frame_n > 0:
                    last_middle = get_middle_v2(boxes_log[frame_n - 1][car])
                    cv2.line(frame, last_middle, middle, (0, 0, 255), 2)
                    distance = round(get_distance(last_middle, middle), 2)
                    if int(distance) == 0:
                        unmoving_cars += 1
                    if unmoving_cars == cars - (cars//8):
                        # Reset tracker
                        re = True
                    # cv2.putText(frame, str(distance), last_middle, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                # cv2.circle(frame, middle, 2, (255, 0, 0), -1)
                car += 1
            frame_n += 1

        # Distance
        frames = len(boxes_log)
        if frames > 1:  # Overall frames
            for old, new in zip(boxes_log[frames - 2], boxes_log[frames - 1]):
                old_middle = get_middle_v2(old)
                new_middle = get_middle_v2(new)
                distance = round(get_distance(old_middle, new_middle), 2)
                area = calc_area(new)
                # t = test_speed(distance, f_start - end_frame, area)
                cv2.putText(frame, str(round(distance, 2)), new_middle, cv2.FONT_HERSHEY_SIMPLEX, 1, (144, 21, 255), 2)

        if fps > 0:
            cv2.putText(frame, str(fps), (5, 29), cv2.FONT_HERSHEY_SIMPLEX, 1, (9, 227, 147), 2)

        # Adding the current frame's boxes to boxes_log
        boxes_log[m] = boxes

        cv2.imshow("Run2712", frame)
        m += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):  # q pressed
            break

        end_frame = time.time()


def test_speed(distance, time_d, area):

    return round(((distance * (1/area)) / time_d) * 10000, 2)  # TODO: Pretty much this little fucker.


def box_to_cords(boxes):
    """
    Parse the boxes so it'd match the cords requirements.
    :param boxes:
    :return:
    """
    new_boxes = []
    for box in boxes:
        new_box = (box[0][0], box[0][1], abs(box[1][0] - box[0][0]), abs(box[1][1] - box[0][1]))
        new_boxes.append(new_box)
    return new_boxes


def main():
    video = r"../Sources/test2.mp4"
    cap = cv2.VideoCapture(video)

    suc, frame = cap.read()
    image_y, boxes = run_yolo(frame, threshold=0.1)  # Process it through Yolo
    cv2.imshow("dada", image_y)
    cv2.waitKey(-1)

    tracker_type = "KCF"

    while cap.isOpened():
        success, frame = cap.read()  # Reads the frame and success

        image_y, boxes = run_yolo(frame, threshold=0.1)  # Process it through Yolo
        boxes = box_to_cords(boxes_to_list(boxes, ["person", "traffic light"]))  # Extract the boxes and proccess it so it'd match the usage
        cv2.imshow("dada", image_y)

        multi_tracker = create_trackers(boxes, tracker_type, frame)  # Creates a tracker for the boxes
        update(multi_tracker, cap)


main()
