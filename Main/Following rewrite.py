import json
import math
import cv2
from download import download_video
from simple_functions import run_yolo, boxes_to_list, get_middle_v2, get_distance, calc_area_v2
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
    s = time.time()
    boxes_log = {}
    dis_for_avrg = {}
    dist_average = None
    end_frame = None
    re = False
    to_break = False
    m = 0
    frame_fps_counter = 0  # FOR FPS
    frame_counter = 0
    fps = 0

    while cap.isOpened():
        success, frame = cap.read()

        f_start = time.time()

        frame_fps_counter += 1  # FPS calculator
        frame_counter += 1  # Frame counter...
        if time.time() - s > 1:
            fps = frame_fps_counter
            s = time.time()
            frame_fps_counter = 0

        if not success:
            break

        # print(frame_counter)

        success, boxes = multi_tracker.update(frame)
        # if re:
        #     to_break = True

        # for new_box, old_box in zip(boxes, boxes_log[m-1]):
        car = 0
        np.random.seed(42)  # Just for the random colors.
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
                        to_break = True

                car += 1
            frame_n += 1

        # Distance
        frames = len(boxes_log)
        if frames > 1:  # Overall frames
            car_counter = 0
            for old, new in zip(boxes_log[frames - 2], boxes_log[frames - 1]):
                old_middle = get_middle_v2(old)
                new_middle = get_middle_v2(new)
                distance = round(get_distance(old_middle, new_middle), 2)

                if frames-2 not in dis_for_avrg:  # For calculating the average distance
                    dis_for_avrg[frames-2] = {}
                dis_for_avrg[frames-2][car_counter] = distance

                # area = calc_area(new)
                # t = test_speed(distance, f_start - end_frame, area)
                cv2.putText(frame, str(distance), new_middle, cv2.FONT_HERSHEY_SIMPLEX, 1, (144, 21, 255), 2)
                car_counter += 1
        if fps > 0:
            cv2.putText(frame, str(fps), (5, 29), cv2.FONT_HERSHEY_SIMPLEX, 1, (9, 227, 147), 2)

        # Adding the current frame's boxes to boxes_log
        boxes_log[m] = boxes

        cv2.imshow("Run2712", frame)
        m += 1

        if frame_counter > 20:  # Break and start again after 20 frames
            break
            # print("20 frames passed")

        if cv2.waitKey(1) & 0xFF == ord('q') or to_break:  # q pressed
            # dist_average = calc_avg(dis_for_avrg)
            break

        end_frame = time.time()
    return dis_for_avrg


def calc_avg(data):
    """
    Calculate the average speed for that frames rate
    :param data:
    :return:
    """
    cars = len(data)
    avgs = []
    # First define average for each car
    for car in data:
        sum_dist = 0
        frames = len(data[car])
        for frame in data[car]:
            sum_dist += data[car][frame]
        avgs.append(sum_dist/frames)
    # Then define average for all cars using the car's average
    overall_avg = sum(avgs)/cars
    return overall_avg


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


def delete_big_box(boxes):
    """
    Litterly as the title declares, deletes the big box in some of the pictures.
    :param boxes:
    :return:
    """
    areas = []
    for vec in boxes:
        for box in boxes[vec]:
            area = math.sqrt(calc_area_v2(box))  # ** 2 so it will remove the extremest
            areas.append(area)

    ar_avg = sum(areas)/len(areas)  # Area Average
    new_boxes = {}
    for vec in boxes:
        for box in boxes[vec]:
            area = math.sqrt(calc_area_v2(box))
            if ar_avg*3 > area:
                if vec not in new_boxes:
                    new_boxes[vec] = []
                new_boxes[vec].append(box)
                # if len(boxes[vec]) > 1:
                #     boxes[vec].remove(box)
                # else:
                #     del boxes[vec]
    return new_boxes


def avg_cars(road, json_path):
    """

    :param road:
    :param json_path:
    :return:
    """
    sum_cars = 0
    with open(json_path, 'r') as f:
        data = json.load(f)
    if road not in data:
        return None
    for video in data[road]:
        sum_cars += data[road][video]["cars"]
    avg = sum_cars/len(data[road])
    return avg


def main(road):
    result_json = r"../results/test1.json"
    video = download_video(road)
    if not video:
        print("Video failed to download, something may be wrong with the site")
        return
    # video = r"C:\Users\עידן\OneDrive - Raanana Schools\Coding\Projects\School Project - Traffic Detection\Examples\JISER\1523-1305.mp4"
    cap = cv2.VideoCapture(video)
    avgs_dist = []
    total_cars = 0
    # print(video)
    road = video.split("\\")[-2]
    file = video.split("\\")[-1].split(".")[0]  # Find a better way to get the name

    # suc, frame = cap.read()
    # image_y, boxes = run_yolo(frame, threshold=0.1)  # Process it through Yolo
    # cv2.imshow("dada", image_y)
    # cv2.waitKey(-1)

    tracker_type = "KCF"

    while cap.isOpened():
        success, frame = cap.read()  # Reads the frame and success
        try:
            image_y, boxes = run_yolo(frame, threshold=0.1)  # Process it through Yolo
        except AttributeError:  # Just for now until I fix that stupid mistake where the video ends and it just suicides
            break
        boxes = delete_big_box(boxes)
        boxes = box_to_cords(boxes_to_list(boxes, ["person", "traffic light"]))  # Extract the boxes and process it so it'd match the usage
        # cv2.imshow("dada", image_y)

        multi_tracker = create_trackers(boxes, tracker_type, frame)  # Creates a tracker for the boxes
        data = update(multi_tracker, cap)
        if not data:
            print("Something failed, continuing")
            continue

        dist_average = calc_avg(data)

        avgs_dist.append(dist_average)
        total_cars += len(data[0])

    if avgs_dist:
        average = sum(avgs_dist)/len(avgs_dist)
        # print("Sum:")
        # print(average)
    else:
        average = 0
        # print(avgs_dist)
    # print(total_cars, "cars")

    average_cars_alltime = avg_cars(road, result_json)
    with open(result_json, 'r') as f:
        data = json.load(f)
        if road not in data:
            data[road] = {}
        if file not in data[road]:
            data[road][file] = {}
        data[road][file]["distance"] = average
        data[road][file]["cars"] = total_cars
    with open(result_json, 'w') as f:
        json.dump(data, f)

    # As simple as that, if he cars are less than the average, the traffic is worse
    if not average_cars_alltime:
        print("There's not enough data.")
    elif total_cars > average_cars_alltime:
        sad_face_path = r"..\Sources\SadFace.png"
        sad_face = cv2.imread(sad_face_path)
        print("The traffic now is worse, you should not drive this road")
        cv2.imshow("Worse the usual", sad_face)
        cv2.waitKey(-1)
    elif total_cars < average_cars_alltime:
        happy_face_path = r"..\Sources\HappyFace.png"
        happy_face = cv2.imread(happy_face_path)
        print("Traffic is better than usual")
        cv2.imshow("Better than usual", happy_face)
        cv2.waitKey(-1)


star = time.time()
main("ALUFSADE")
print(time.time() - star)
