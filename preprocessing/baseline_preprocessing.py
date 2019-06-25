import h5py
import sys
import numpy as np


def remove_element(array, element):
    return array[array != element]


source_file_path = sys.argv[1]
target_file_path = sys.argv[2]

source_data = h5py.File(source_file_path, 'r')
target_data = h5py.File(target_file_path, 'w')
target_data.create_dataset("itemID", data=source_data['itemID'][:].tolist())
target_data.create_dataset('image_id', data=source_data['image_id'][:].tolist())

source_descriptions = source_data['descriptions'][:].tolist()
target_descriptions = []
print(source_descriptions[0])
elements = [0, 1, 2]

for desc in source_descriptions:
    array = np.array(desc)
    for elem in elements:
        array = remove_element(array, elem)

    target_descriptions.append(array)

target_data.create_dataset("descriptions", data=target_descriptions)