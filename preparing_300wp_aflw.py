from utils.data_utils.data_preparing_utils import create_labels_json
import pandas as pd
import numpy as np

data = pd.read_csv("../../Datasets/train_test_split_ALL.csv")

create_labels_json(data, dataset=['AFLW2000'])