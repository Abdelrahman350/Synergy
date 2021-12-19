import cv2
import numpy as np
from numpy import cos, sin
from data_generator.labels_preprocessing import denormalize, param3DMM_to_pose

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
    end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
    pt2d = np.round(pt2d).astype(np.int32)
    for i in range(pt2d.shape[0]):
        start_point = pt2d[i]
        cv2.circle(image, (start_point[0], start_point[1]), 2, (0, 0, 1), -1)
        if i in end_list:
            continue
        end_point = pt2d[i+1]
        cv2.line(image, (start_point[0], start_point[1]),\
             (end_point[0], end_point[1]), (1, 1, 1), 1)
    return image

def plot_pose(image, label, name='output_axis'):
    label = denormalize(label)
    pitch, yaw, roll = param3DMM_to_pose(label[:12])
    image = draw_axis(image, pitch, yaw, roll)     
    cv2.imwrite('output/'+name+".jpg", image*255)

def plot_landmarks(image, pt2d, name='output_landmarks'):
    image = draw_landmarks(image, pt2d)
    cv2.imwrite('output/'+name+'.jpg', image*255)