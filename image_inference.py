import glob
from data_generator.image_preprocessing import crop_img, normalization, resize_image
from data_generator.labels_preprocessing import denormalize, param3DMM_to_pose
from set_tensorflow_configs import set_GPU
from utils.data_utils.plotting_data import draw_axis, plot_landmarks, plot_pose
from model.synergy import Synergy
import argparse
import numpy as np
import cv2
from os import path
from utils.face_detection_utils import infer
from utils.inference_utils import predict_lmks
from utils.openvino_utils import openvino

set_GPU()

def main(args):
    IMG_H = 128
    input_shape = (IMG_H, IMG_H, 3)
    model_path = "checkpoints/Synergy_300W_AFLW_mse"
    model = Synergy(input_shape=input_shape, morphable='pca')

    print(model.summary())
    model.load_weights(model_path).expect_partial()
    exec_net, input_key = openvino('640_640/model_int8/optimized/scrfd.bin')

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
            rect = np.concatenate((rect[2:], rect[1:2]))
            roi_box = rect[0:4]
            HCenter = (rect[1] + rect[3])/2
            WCenter = (rect[0] + rect[2])/2
            height_len = roi_box[3]-roi_box[1]
            width_len = roi_box[2]-roi_box[0]
            margin_h = height_len * 1.2 // 2
            margin_w = width_len * 1.2 // 2
            roi_box[0] = WCenter-margin_w
            roi_box[1] = HCenter-margin_h*1.25
            roi_box[2] = WCenter+margin_w
            roi_box[3] = HCenter+margin_h

            image = crop_img(img_ori, roi_box)
            image, aspect_ratio = resize_image(image, input_shape)
            image = normalization(image)

            image = np.expand_dims(image, axis=0)
            prediction = model.predict(image)
            param_3DMM = prediction['Pm']
            
            lmks = predict_lmks(param_3DMM, roi_box)
            
            pose = denormalize(param_3DMM)
            angels = param3DMM_to_pose(pose[:, :12])
            vertices.append(lmks)
            poses.append(angels)

        name = img_fp.rsplit('/', 1)[-1][:-4]
        image_vertices = img_ori.copy()
        image_vertices = normalization(image_vertices)
        image_vertices = plot_landmarks(image_vertices, vertices)
        cv2.imwrite(name+'_lmks.png', image_vertices)
        image_poses = img_ori.copy()
        image_poses = normalization(image_poses)
        for i in range(len(poses)):
            pose = poses[i]
            image_poses  = draw_axis(image_poses, pose[0], pose[1], pose[2], vertices[i])
        image_poses += 1.0
        image_poses *= 127.5
        print(image_poses.max(), image_poses.min())
        cv2.imwrite(name+'_poses.png', image_poses)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', default='',\
         help='path to a single image or path to a folder containing multiple images')
    args = parser.parse_args()
    main(args)