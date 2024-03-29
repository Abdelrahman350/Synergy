import cv2
import numpy as np
from numpy import cos, sin
from data_generator.image_preprocessing import denormalize_image

def draw_axis(image_original, pitch, yaw, roll, pt2d):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    image = image_original.copy()
    pitch, yaw, roll = pitch, -yaw, roll

    tdx = pt2d[30, 0]
    tdy = pt2d[30, 1]
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = image.shape[:2]
        tdx = width / 2
        tdy = height / 2
    
    minx, maxx = np.min(pt2d[:, 0]), np.max(pt2d[:, 0])
    miny, maxy = np.min(pt2d[:, 1]), np.max(pt2d[:, 1])
    llength = np.sqrt((maxx - minx) * (maxy - miny))
    size = llength * 0.5

    # X-Axis pointing to right, drawn in red.
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis pointing downward, drawn in green. 
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis pointing out of the screen, drawn in blue.
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(image, (int(tdx), int(tdy)), (int(x1),int(y1)), (-1,-1,1), 2)
    cv2.line(image, (int(tdx), int(tdy)), (int(x2),int(y2)), (-1,1,-1), 2)
    cv2.line(image, (int(tdx), int(tdy)), (int(x3),int(y3)), (1,-1,-1), 2)
    return image

def draw_landmarks(image_original, pt2d):
    image = image_original.copy()
    end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
    pt2d = np.round(pt2d).astype(np.int32)
    for i in range(pt2d.shape[0]):
        start_point = pt2d[i]
        cv2.circle(image, (start_point[0], start_point[1]), 3, (-1,-1,1), -1)
        if i in end_list:
            continue
        end_point = pt2d[i+1]
        cv2.line(image, (start_point[0], start_point[1]),\
             (end_point[0], end_point[1]), (1, 1, 1), 1)
    return image

def plot_pose(image, theta, pt2d):
    if np.ndim(theta) == 1 and theta:
        pitch, yaw, roll = theta
        image = draw_axis(image, pitch, yaw, roll, pt2d)
    elif np.ndim(theta) == 2:
        for i in range(np.shape(theta)[0]):
            pitch, yaw, roll = theta[i]
            image = draw_axis(image, pitch, yaw, roll, pt2d[i])
    image = denormalize_image(image)
    return image

def plot_landmarks(image, pt2d):
    if np.ndim(pt2d) == 2:
        image = draw_landmarks(image, pt2d)
    elif np.ndim(pt2d) == 3:
        for i in range(np.shape(pt2d)[0]):
            image = draw_landmarks(image, pt2d[i])
    image = denormalize_image(image)
    return image