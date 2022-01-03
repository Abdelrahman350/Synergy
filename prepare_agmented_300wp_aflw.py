from utils.data_utils.data_preparing_utils import dictionary_to_json
from utils.loading_data_augmented import get_IDs, get_labels, load

IDs = get_IDs()
labels = get_labels()
dictionary_to_json(IDs, '../../Datasets/300W_AFLW_Augmented/IDs_augmented')
dictionary_to_json(labels, '../../Datasets/300W_AFLW_Augmented/labels_augmented')