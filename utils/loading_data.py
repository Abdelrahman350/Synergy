import json
from data_generator.data_generator import DataGenerator

def loading_dictionaries(dataset='300w'):
    base_dir_ids = "../../Datasets/partition_"
    base_dir_labels = "../../Datasets/labels_"
    json_file_ids = json_file_labels = 'None'
    if dataset=='300w':
        json_file_ids = base_dir_ids + 'LP.json'
        json_file_labels = base_dir_labels + 'LP.json'
    elif dataset=='AFLW':
        json_file_ids = base_dir_ids + 'AFLW.json'
        json_file_labels = base_dir_labels + 'AFLW.json'
    print(json_file_ids)
    with open(json_file_ids, 'r') as j:
        partition = json.loads(j.read())
    with open(json_file_labels, 'r') as j:
        labels = json.loads(j.read())
    return partition, labels

def loading_generators(dataset='300w', input_shape=(224, 224, 3), batch_size=16, shuffle=True):
    partition, labels = loading_dictionaries(dataset=dataset)
    training_data_generator = DataGenerator(partition['train'], labels,\
        batch_size=batch_size, input_shape=input_shape, shuffle=shuffle)

    validation_data_generator = DataGenerator(partition['valid'], labels,\
        batch_size=batch_size, input_shape=input_shape, shuffle=shuffle)
    return training_data_generator, validation_data_generator