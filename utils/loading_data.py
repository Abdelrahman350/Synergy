import json
from data_generator.data_generator import DataGenerator, DataGenerator_DDFA

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
        json_file_ids = base_dir_ids + 'DDFA.json'
        json_file_labels = base_dir_labels + 'DDFA.json'
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
        partition_train, labels_train = loading_dictionaries(dataset='DDFA')
        training_data_generator = DataGenerator_DDFA(partition_train, labels_train,\
            batch_size=batch_size, input_shape=input_shape, shuffle=shuffle,\
                dataset_path='../../Datasets/300W_AFLW_Augmented/')

        partition_valid, labels_valid = loading_dictionaries(dataset='AFLW')
        partition_combined_valid = partition_valid['train'] + partition_valid['valid']
        validation_data_generator = DataGenerator(partition_combined_valid, labels_valid,\
            batch_size=batch_size, input_shape=input_shape, shuffle=shuffle)
    else:    
        partition, labels = loading_dictionaries(dataset=dataset)
        training_data_generator = DataGenerator(partition, labels, batch_size=batch_size,\
             input_shape=input_shape, shuffle=shuffle)

        validation_data_generator = DataGenerator(partition['valid'], labels,\
            batch_size=batch_size, input_shape=input_shape, shuffle=shuffle)
    return training_data_generator, validation_data_generator

def loading_test_examples(test, input_shape):
    if test == 'AFLW':
        training_data_generator, validation_data_generator = loading_generators(dataset='AFLW',\
                input_shape=input_shape, batch_size=32, shuffle=True)
        list_ids = [
            "AFLW2000-3D/AFLW2000/image01986.jpg", "AFLW2000-3D/AFLW2000/image00405.jpg", 
            "AFLW2000-3D/AFLW2000/image02650.jpg", "AFLW2000-3D/AFLW2000/image00291.jpg", 
            "AFLW2000-3D/AFLW2000/image02522.jpg", "AFLW2000-3D/AFLW2000/image04269.jpg", 
            "AFLW2000-3D/AFLW2000/image03515.jpg", "AFLW2000-3D/AFLW2000/image02183.jpg", 
            "AFLW2000-3D/AFLW2000/image04102.jpg", "AFLW2000-3D/AFLW2000/image01079.jpg", 
            "AFLW2000-3D/AFLW2000/image00187.jpg", "AFLW2000-3D/AFLW2000/image00359.jpg",
            "AFLW2000-3D/AFLW2000/image04188.jpg", "AFLW2000-3D/AFLW2000/image02243.jpg", 
            "AFLW2000-3D/AFLW2000/image00053.jpg", "AFLW2000-3D/AFLW2000/image02213.jpg", 
            "AFLW2000-3D/AFLW2000/image04004.jpg", "AFLW2000-3D/AFLW2000/image03043.jpg", 
            "AFLW2000-3D/AFLW2000/image01366.jpg", "AFLW2000-3D/AFLW2000/image02782.jpg", 
            "AFLW2000-3D/AFLW2000/image02664.jpg",
            "AFLW2000-3D/AFLW2000/image00741.jpg", "AFLW2000-3D/AFLW2000/image00771.jpg", 
            "AFLW2000-3D/AFLW2000/image00062.jpg", 
            "AFLW2000-3D/AFLW2000/image00554.jpg", "AFLW2000-3D/AFLW2000/image03077.jpg", 
            "AFLW2000-3D/AFLW2000/image03705.jpg", 
            "AFLW2000-3D/AFLW2000/image02597.jpg", "AFLW2000-3D/AFLW2000/image01981.jpg", 
            "AFLW2000-3D/AFLW2000/image02918.jpg", "AFLW2000-3D/AFLW2000/image03640.jpg", 
            "AFLW2000-3D/AFLW2000/image01427.jpg", 
            "AFLW2000-3D/AFLW2000/image01449.jpg", "AFLW2000-3D/AFLW2000/image00922.jpg", 
            "AFLW2000-3D/AFLW2000/image03375.jpg", 
            "AFLW2000-3D/AFLW2000/image01688.jpg", "AFLW2000-3D/AFLW2000/image02038.jpg", 
            "AFLW2000-3D/AFLW2000/image03479.jpg", 
            "AFLW2000-3D/AFLW2000/image01110.jpg", "AFLW2000-3D/AFLW2000/image03897.jpg", 
            "AFLW2000-3D/AFLW2000/image01649.jpg", 
            "AFLW2000-3D/AFLW2000/image02598.jpg", "AFLW2000-3D/AFLW2000/image00809.jpg", 
            "AFLW2000-3D/AFLW2000/image00060.jpg"]
    elif test == '300W':
        training_data_generator, validation_data_generator = loading_generators(dataset='300W',\
                input_shape=input_shape, batch_size=32, shuffle=True)
        list_ids = ["300W-LP/300W_LP/AFW/AFW_134212_1_2.jpg",
            "300W-LP/300W_LP/HELEN_Flip/HELEN_1269874180_1_0.jpg", 
            "300W-LP/300W_LP/AFW/AFW_4512714865_1_3.jpg",
            "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0737_13.jpg",
            "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0047_4.jpg"]
    elif test == 'DDFA':
        training_data_generator, validation_data_generator = loading_generators(dataset='DDFA',\
                input_shape=input_shape, batch_size=32, shuffle=True)
        list_ids = ["train_aug_120x120/LFPWFlip_LFPW_image_train_0859_12_3.jpg",
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_12_2.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_12_1.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_13_3.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_13_2.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_13_1.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_14_2.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_14_1.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_15_2.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_15_1.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_16_2.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_16_1.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_17_2.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_17_1.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_1_11.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_1_10.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_1_9.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_1_8.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_1_7.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_1_6.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_1_5.jpg", 
            "train_aug_120x120/LFPWFlip_LFPW_image_train_0859_1_4.jpg"]
    return list_ids, training_data_generator, validation_data_generator