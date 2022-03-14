import glob
import os
from data_generator.image_preprocessing import crop_img, normalize_image, resize_image
from data_generator.labels_preprocessing import denormalize_param, Param3D_to_Pose
from set_tensorflow_configs import set_GPU
from utils.data_utils.plotting_data import plot_landmarks, plot_pose
from model.synergy import Synergy
import argparse
import numpy as np
import cv2
from os import path
from os.path import join
from utils.face_detection_utils import infer
from utils.inference_utils import PCA, predict_lmks
from utils.openvino_utils import openvino

set_GPU()

def main(args):
    IMG_H = 128
    input_shape = (IMG_H, IMG_H, 3)
    model_path = "checkpoints/Synergy/mobileNetV2_MSE"
    plot_bboxes = False

    exec_net, input_key = openvino('640_640/model_int8/optimized/scrfd.bin')
    model = Synergy(input_shape=input_shape, morphable='PCA')
    print(model.summary())
    model.load_weights(model_path).expect_partial()

    if path.isdir(args.files):
        if not args.files[-1] == '/':
            args.files = args.files + '/'
        if not args.png:
            files = sorted(glob.glob(args.files+'*.jpg'))
        else:
            files = sorted(glob.glob(args.files+'*.png'))
    else:
        files = [args.files]

    for img_fp in files:
        print("Process the image: ", img_fp)
        img_ori = cv2.imread(img_fp)
        bboxes = infer(img_ori, exec_net, input_key, 0.5)
        # storage
        vertices = []
        poses = []

        for idx, rect in enumerate(bboxes):
            if plot_bboxes:
                _,score,x1,y1,x2,y2 = rect.astype(np.int)
                cv2.rectangle(img_ori, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
            rect = np.concatenate((rect[2:], rect[1:2]))
            roi_box = rect[0:4]
            HCenter = (rect[1] + rect[3])/2
            WCenter = (rect[0] + rect[2])/2
            side_len = roi_box[3]-roi_box[1]
            margin = side_len * 1.2 // 2
            roi_box[0] = WCenter-margin
            roi_box[1] = HCenter-margin
            roi_box[2] = WCenter+margin
            roi_box[3] = HCenter+margin
            
            image = crop_img(img_ori, roi_box)
            image, aspect_ratio = resize_image(image, input_shape)
            image = normalize_image(image)
            image = np.expand_dims(image, axis=0)
            
            prediction = model.predict(image)
            param_3DMM = prediction['Pm']
            lmks = np.squeeze(PCA(input_shape=input_shape)(param_3DMM), 0) * 1/aspect_ratio
            lmks[:, 0] += roi_box[0]
            lmks[:, 1] += roi_box[1]

            pose = denormalize_param(param_3DMM)
            angels = Param3D_to_Pose(pose[:, :12])
            vertices.append(lmks)
            poses.append(angels)

        name = img_fp.rsplit('/', 1)[-1][:-4]
        image_vertices = img_ori.copy()
        image_vertices = normalize_image(image_vertices)
        image_vertices = plot_landmarks(image_vertices, vertices)
        
        image_poses = img_ori.copy()
        image_poses = normalize_image(image_poses)
        image_poses = plot_pose(image_poses, poses, vertices)

        lmks_output_path = 'image_inference_output/landmarks/'
        poses_output_path = 'image_inference_output/poses/'
        if not path.exists(f'image_inference_output/'):
            os.makedirs(f'image_inference_output/')
        if not path.exists(lmks_output_path):
            os.makedirs(lmks_output_path)
        if not path.exists(poses_output_path):
            os.makedirs(poses_output_path)
        
        cv2.imwrite(join(lmks_output_path, name+'.jpg'), image_vertices)
        cv2.imwrite(join(poses_output_path, name+'.jpg'), image_poses)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', default='',\
         help='path to a single image or path to a folder containing multiple images')
    parser.add_argument("--png", action="store_true", help="if images are with .png extension")
    args = parser.parse_args()
    main(args)