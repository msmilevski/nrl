import numpy as np
import pickle
import h5py

dict = pickle.load(open('dataset/Image_embed_dict.pickle', 'rb'))

for item in dict.keys():
    data = h5py.File("dataset/resnet152/image_features_" + str(item) + ".hdf5", 'r')
    data_ids = data['image_id'][()]
    img_ids = dict[item]
    img_embeddings = []
    for img_id in img_ids:
        position_item = np.argwhere(data_ids == img_id)[0][0]
        img_embeddings.append(data['image_features'][position_item])

    save_file_path = "/home/s1885778/nrl/dataset/resnet152_1/image_features_" + str(item) + ".hdf5"
    print("Saving file: " + save_file_path + " ...")
    data_file = h5py.File(save_file_path, 'w')
    data_file.create_dataset("image_id", data=img_ids)
    data_file.create_dataset("image_features", data=img_embeddings)