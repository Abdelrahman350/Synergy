from utils.data_utils.data_preparing_utils import create_labels_json
import pandas as pd

data = pd.read_csv("../../Datasets/300W_AFLW/train_test_split_ALL.csv")

create_labels_json(data, dataset=['300W_LP'])