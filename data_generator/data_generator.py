import tensorflow as tf
import numpy as np
from data_generator.image_preprocessing import image_loader
from data_generator.labels_preprocessing import label_loader

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, input_shape=(128, 128, 3),
                 shuffle=True, dataset_path='../../Datasets/'):
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.dataset_path = dataset_path
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
        batch_parameters_3DMM = []
        for index, image_id in enumerate(batch):
            image, aspect_ratio = image_loader(image_id, self.dataset_path, self.input_shape)
            X.append(image)
            parameters_3DDM = label_loader(image_id, self.labels, aspect_ratio)
            batch_parameters_3DMM.append(parameters_3DDM)
        X = np.array(X)
        batch_parameters_3DMM = np.array(batch_parameters_3DMM)
        return X, batch_parameters_3DMM
    
    def data_generation(self, batch):
        # Initializing input data
        X = []
        batch_parameters_3DMM = []
        for index, image_id in enumerate(batch):
            image, aspect_ratio = image_loader(image_id, self.dataset_path, self.input_shape)
            X.append(image)
            parameters_3DDM = label_loader(image_id, self.labels, aspect_ratio)
            batch_parameters_3DMM.append(parameters_3DDM)
        X = np.array(X)
        batch_parameters_3DMM = np.array(batch_parameters_3DMM)
        pose_3DMM = batch_parameters_3DMM[:, :12]
        alpha_exp = batch_parameters_3DMM[:, 12:22]
        alpha_shp = batch_parameters_3DMM[:, 22:]
        return X, (pose_3DMM, alpha_exp, alpha_shp)

    def get_one_instance(self, id):
        batch = [id]
        X, y = self.__data_generation(batch)
        return X[0], y[0]