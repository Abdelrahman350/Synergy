import cv2
from tensorflow.keras.utils import Sequence
import numpy as np
from data_generator.image_preprocessing import colorjitter, crop, filters, gray_img, noisy, normalization, resize_image
from data_generator.labels_preprocessing import label_loader, normalize
from model.morhaple_face_model import PCA
from os.path import join

class DataGenerator(Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, input_shape=(128, 128, 3),
                 shuffle=True, dataset_path='../../Datasets/300W_AFLW/'):
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.dataset_path = dataset_path
        self.pca = PCA((450, 450, 3))
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
            image_path = join(self.dataset_path, image_id+'.jpg')
            image = cv2.imread(image_path)
            parameters_3DMM = label_loader(image_id, self.labels)
            parameters_3DMM = normalize(parameters_3DMM)
            pt3d = self.pca(np.expand_dims(parameters_3DMM, 0))
            image, roi_box = crop(image, pt3d)
            image, aspect_ratio = resize_image(image, self.input_shape)
            # Augment image
            aug_type = np.random.choice(['color', 'gray', 'None'])
            if aug_type == 'color':
                image = colorjitter(image)
                image = noisy(image)
                image = filters(image)
            elif aug_type == 'gray':
                image = gray_img(image)
            image = normalization(image)
            # Label preprocessing
            sx, sy, ex, ey = roi_box
            parameters_3DMM = label_loader(image_id, self.labels)
            parameters_3DMM[3] = parameters_3DMM[3] - sx
            parameters_3DMM[7] = parameters_3DMM[7] + sy
            parameters_3DMM = normalize(parameters_3DMM)
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