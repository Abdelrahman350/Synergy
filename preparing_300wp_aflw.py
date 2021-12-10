from utils.data_utils.data_preparing_utils import get_IDs, dictionary_to_json, get_labels
from utils.data_utils import label_parameters
import pandas as pd

data = pd.read_csv("../../Datasets/train_test_split_ALL.csv")
print(label_parameters.get_pose_from_mat("../../Datasets/300W-LP/300W_LP/AFW/AFW_134212_1_2.mat"))

