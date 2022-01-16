import json
from data_generator.data_generator import DataGenerator

def loading_dictionaries(dataset='300w'):
    base_dir_ids = base_dir_labels = 'None'
    json_file_ids = json_file_labels = 'None'
    if dataset=='300W':
        base_dir_ids = "../../Datasets/300W_AFLW/IDs_"
        base_dir_labels = "../../Datasets/300W_AFLW/labels_"
        json_file_ids = base_dir_ids + '300W_LP.json'
        json_file_labels = base_dir_labels + '300W_LP.json'
    elif dataset=='AFLW':
        base_dir_ids = "../../Datasets/300W_AFLW/IDs_"
        base_dir_labels = "../../Datasets/300W_AFLW/labels_"
        json_file_ids = base_dir_ids + 'AFLW2000.json'
        json_file_labels = base_dir_labels + 'AFLW2000.json'
    elif dataset=='DDFA':
        base_dir_ids = "../../Datasets/300W_AFLW_Augmented/IDs_"
        base_dir_labels = "../../Datasets/300W_AFLW_Augmented/labels_"
        json_file_ids = base_dir_ids + 'AFLW2000.json'
        json_file_labels = base_dir_labels + 'AFLW2000.json'
    with open(json_file_ids, 'r') as j:
        IDs = json.loads(j.read())
    with open(json_file_labels, 'r') as j:
        labels = json.loads(j.read())
    return IDs, labels

def loading_generators(dataset='300W', input_shape=(224, 224, 3), batch_size=16, shuffle=True):
    if dataset=='all':
        partition_train, labels_train = loading_dictionaries(dataset='300W')
        partition_combined_train = partition_train['train'] + partition_train['valid']
        training_data_generator = DataGenerator(partition_combined_train, labels_train,\
            batch_size=batch_size, input_shape=input_shape, shuffle=shuffle)
        
        partition_valid, labels_valid = loading_dictionaries(dataset='AFLW')
        partition_combined_valid = partition_valid['train'] + partition_valid['valid']
        validation_data_generator = DataGenerator(partition_combined_valid, labels_valid,\
            batch_size=batch_size, input_shape=input_shape, shuffle=shuffle)
    elif dataset=='DDFA':
        partition_train, labels_train = loading_dictionaries(dataset='300W')
        training_data_generator = DataGenerator(partition_train, labels_train,\
            batch_size=batch_size, input_shape=input_shape, shuffle=shuffle)

        partition_valid, labels_valid = loading_dictionaries(dataset='AFLW')
        partition_combined_valid = partition_valid['train'] + partition_valid['valid']
        validation_data_generator = DataGenerator(partition_combined_valid, labels_valid,\
            batch_size=batch_size, input_shape=input_shape, shuffle=shuffle)
    else:    
        partition, labels = loading_dictionaries(dataset=dataset)
        training_data_generator = DataGenerator(partition['train'], labels,\
            batch_size=batch_size, input_shape=input_shape, shuffle=shuffle)

        validation_data_generator = DataGenerator(partition['valid'], labels,\
            batch_size=batch_size, input_shape=input_shape, shuffle=shuffle)
    return training_data_generator, validation_data_generator