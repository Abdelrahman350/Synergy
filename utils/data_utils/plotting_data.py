import cv2
from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt

def draw_axis(img, pitch, yaw, roll, tdx=None, tdy=None, size = 100):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    pitch, yaw, roll = pitch, -yaw, roll
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
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

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)), (0,0,1), 2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)), (0,1,0), 2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)), (1,0,0), 2)
    return img

def draw_landmarks(img_ori, pts, size=1):
    img = img_ori.copy()
    n = pts.shape[1]
    if n <= 106:
        for i in range(n):
            cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, (0,1,0), -1)
    else:
        sep = 1
        for i in range(0, n, sep):
            cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, (0,1,0), 1)

def plot_pose_image(image, gt_poses):
    pitch, yaw, roll = gt_poses
    image = draw_axis(image, pitch, yaw, roll, tdx=None, tdy=None, size = 100)        
    cv2.imwrite(f"output.jpg", image*255)

def plot_landmarks_image(image, pt2d):
    image = draw_landmarks(image, pt2d, tdx=None, tdy=None, size = 100)        
    cv2.imwrite(f"output.jpg", image*255)