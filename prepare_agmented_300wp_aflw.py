from utils.data_utils.data_preparing_utils import dictionary_to_json
from utils.loading_data_augmented import get_IDs

IDs = get_IDs()

dictionary_to_json(IDs, '../../Datasets/300W_AFLW_Augmented/IDs_augmented.json')