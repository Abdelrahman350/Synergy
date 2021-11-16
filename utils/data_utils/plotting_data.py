import cv2
import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from data_generator.preprocessing_labels import label_3DDm_to_pose, label_3DDm_to_pt2d

def draw_axis(image_original, pitch, yaw, roll, tdx=None, tdy=None, size = 100):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    image = image_original.copy()
    pitch, yaw, roll = pitch, -yaw, roll
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = image.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right, drawn in red.
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis pointing downward, drawn in green. 
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis pointing out of the screen, drawn in blue.
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(image, (int(tdx), int(tdy)), (int(x1),int(y1)), (0,0,1), 2)
    cv2.line(image, (int(tdx), int(tdy)), (int(x2),int(y2)), (0,1,0), 2)
    cv2.line(image, (int(tdx), int(tdy)), (int(x3),int(y3)), (1,0,0), 2)
    return image

def draw_landmarks(image_original, pt2d):
    image = image_original.copy()
    for point in pt2d:
        cv2.circle(image, (round(int(point[0])), round(int(point[1]))), 2, (0, 0, 1), -1)
    return image

def plot_pose_image(image, label):
    pitch, yaw, roll = label_3DDm_to_pose(label)
    image = draw_axis(image, pitch, yaw, roll)        
    cv2.imwrite(f"output_axis.jpg", image*255)

def plot_landmarks_image(image, label):
    pt2d = label_3DDm_to_pt2d(label)
    image = draw_landmarks(image, pt2d)        
    cv2.imwrite(f"output_landmarks.jpg", image*255)