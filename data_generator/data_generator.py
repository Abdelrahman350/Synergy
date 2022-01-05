import pickle
from tensorflow.keras.utils import Sequence
import numpy as np
from data_generator.image_preprocessing import augment, image_loader
from data_generator.labels_preprocessing import label_loader
from model.morhaple_face_model import PCA
from pathlib import Path

class DataGenerator(Sequence):
    def __init__(self, root, filelists, param_fp, gt_transform=False, transform=None, batch_size=8):
        self.root = root
        self.lines = Path(filelists).read_text().strip().split('\n')
        self.params = np.array(_load(param_fp))
        self.gt_transform = gt_transform
        self.indices = np.arange(len(self.lines))
        self.batch_size = batch_size
        
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
            image = image_loader(image_id, self.dataset_path)
            parameters_3DMM = label_loader(image_id, self.labels)
            lmks = self.pca(np.expand_dims(parameters_3DMM, 0))
            # image = augment(image, lmks, self.input_shape)
            X.append(image)
            batch_parameters_3DMM.append(parameters_3DMM)
        X = np.array(X)
        batch_parameters_3DMM = np.array(batch_parameters_3DMM)
        Lc = self.pca(batch_parameters_3DMM)
        return X, {'Pm':batch_parameters_3DMM, 'Pm*':batch_parameters_3DMM, 'Lc':Lc, 'Lr':Lc}

    def data_generation(self, batch):
        # Initializing input data
        X = []
        batch_parameters_3DMM = []
        for index, image_id in enumerate(batch):
            image = image_loader(image_id, self.dataset_path)
            parameters_3DMM = label_loader(image_id, self.labels)
            lmks = self.pca(np.expand_dims(parameters_3DMM, 0))
            # image = augment(image, lmks, self.input_shape)
            X.append(image)
            batch_parameters_3DMM.append(parameters_3DMM)
        X = np.array(X)
        batch_parameters_3DMM = np.array(batch_parameters_3DMM)
        Lc = self.pca(batch_parameters_3DMM)
        return X, {'Pm':batch_parameters_3DMM, 'Pm*':batch_parameters_3DMM, 'Lc':Lc, 'Lr':Lc}

    def get_one_instance(self, id):
        batch = [id]
        X, y = self.__data_generation(batch)
        return X[0], y[0]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]