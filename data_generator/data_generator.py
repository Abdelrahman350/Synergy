import cv2
from tensorflow.keras.utils import Sequence
import numpy as np
from data_generator.image_preprocessing import colorjitter, crop, filters, gray_img, normalize_image
from data_generator.image_preprocessing import noisy, resize_image
from data_generator.labels_preprocessing import denormalize_param, label_loader, normalize_param
from data_generator.labels_preprocessing import eulerAngles_to_RotationMatrix
from model.morhaple_face_model import PCA, Reconstruct_Vertex
from os.path import join

class DataGenerator(Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, input_shape=(128, 128, 3),
                 shuffle=True, augmentation=True, dataset_path='../../Datasets/300W_AFLW/'):
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.dataset_path = dataset_path
        self.pca = PCA((450, 450, 3)) if \
            dataset_path=='../../Datasets/300W_AFLW/' else Reconstruct_Vertex((120, 120, 3))
        self.indices = np.arange(len(self.list_IDs))
        
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def on_epoch_end(self):
        """Shuffle after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    def __getitem__(self, index):
        """Get a batch of data"""
        start_index = index * self.batch_size
        end_index = (index+1) * self.batch_size
        batch_IDs = self.indices[start_index : end_index]
        batch = [self.list_IDs[k] for k in batch_IDs]
        X, y = self.__data_generation(batch)
        return X, y
    
    def __data_generation(self, batch):
        # Initializing input data
        X = []
        Lc = []
        batch_parameters_3DMM = []
        for index, image_id in enumerate(batch):
            image_path = join(self.dataset_path, image_id)
            image = cv2.imread(image_path)

            # Image roll rotation augmentation
            roll_angle = np.random.randint(low=-360, high=360)
            theta_aug = np.array([0, 0, roll_angle]) * np.pi/180
            R_aug = eulerAngles_to_RotationMatrix(theta_aug)
            (h, w) = image.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            # rotate our image by theta degrees around the center in the roll direction
            M = cv2.getRotationMatrix2D((cX, cY), -theta_aug[2]*180/np.pi, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
            parameters_3DMM = label_loader(image_id, self.labels)
            parameters_3DMM = denormalize_param(parameters_3DMM)
            P = parameters_3DMM[:12].reshape((3, 4))
            R_rot = P[:, 0:3]
            t_rot = P[:, 3]
            R_new = R_aug.dot(R_rot)
            t = np.array([
                cX*(1-np.cos(theta_aug[2]))+cY*np.sin(theta_aug[2]),
                cY*(1-np.cos(theta_aug[2]))-cX*np.sin(theta_aug[2]),
                0
            ])
            t_new = R_aug.dot(t_rot.T-t)
            P[:, 0:3] = R_new
            P[:, 3] = t_new
            parameters_3DMM[:12] = P.reshape((-1, 12))
            parameters_3DMM = normalize_param(parameters_3DMM)
            
            # Face crop augmentation
            pt3d = self.pca(np.expand_dims(parameters_3DMM, 0))
            image, roi_box = crop(image, pt3d)
            image, aspect_ratio = resize_image(image, self.input_shape)
            # Label preprocessing
            sx, sy, _, _ = roi_box
            parameters_3DMM[3] = parameters_3DMM[3] - sx/self.pca.param_std[3]
            parameters_3DMM[7] = parameters_3DMM[7] + sy/self.pca.param_std[7]
            
            # Augment image color and illumination
            if self.augmentation:
                aug_type = np.random.choice(['color', 'gray', 'None'])
                if aug_type == 'color':
                    image = colorjitter(image)
                    image = noisy(image)
                    image = filters(image)
                elif aug_type == 'gray':
                    image = gray_img(image)

            image = normalize_image(image)
            lmks = self.pca(np.expand_dims(parameters_3DMM, 0)).numpy() *  aspect_ratio
            X.append(image)
            Lc.append(np.squeeze(lmks, axis=0))
            batch_parameters_3DMM.append(parameters_3DMM)
        X = np.array(X)
        batch_parameters_3DMM = np.array(batch_parameters_3DMM)
        Lc = np.array(Lc)
        return X, {'Pm':batch_parameters_3DMM, 'Pm*':batch_parameters_3DMM, 'Lc':Lc, 'Lr':Lc}

    def data_generation(self, batch):
        return self.__data_generation(batch)

    def get_one_instance(self, id):
        batch = [id]
        X, y = self.__data_generation(batch)
        return X[0], y[0]