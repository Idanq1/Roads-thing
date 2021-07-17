import ffmpeg
import datetime
import time
import json
import os

roads_path = r"..\data\roads.json"
with open(roads_path, 'r') as f:
    playlist = json.load(f)


def download_video(road, path=None):
    if not path:
        if road not in os.listdir(r"..\Examples"):
            os.mkdir(f"..\\Examples\\{road}")
        path = f"..\\Examples\\{road}"
    now_time = datetime.datetime.now()
    hour = add_0_start(now_time.hour)
    minute = add_0_start(now_time.minute)
    if len(minute) < 2:
        minute = '0' + minute
    day = add_0_start(now_time.day)  # Day of date (*27*/5)
    month = add_0_start(now_time.month)  # In numbers
    video_name = f"{path}\\{hour}{minute}-{day}{month}.mp4"
    try:
        (ffmpeg.input(playlist[road], t=10).output(video_name).run())
        return video_name
    except:
        return


def add_0_start(num):
    """
    Adds 0 at the start of the number, for example: We don't want the hour to be 182, we want it to be 1802, so we add a 0 at the start of the number
    :param num:
    :type num: int
    :return:
    :rtype str:
    """
    num = str(num)
    if len(num) < 2:
        return '0' + num
    return num


if __name__ == '__main__':
    download_video("ALUFSADE")
