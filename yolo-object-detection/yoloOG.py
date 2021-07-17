import numpy as np
import argparse
import imutils
import time
import cv2
import os

coco_path = "D:\\Python\\cv2\\yolo-object-detection\\yolo-coco\\coco.names"

labels = open(coco_path).read().strip().split("\n")

# np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
