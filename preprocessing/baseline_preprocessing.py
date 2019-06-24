import h5py
import sys


def remove_element(array, element):
    return array[array != element]


source_file_path = sys.argv[1]
target_file_path = sys.argv[2]

source_data = h5py.File(source_file_path, 'r')
target_data = h5py.File(target_file_path, 'w')
target_data.create_dataset("itemID", data=source_data['itemID'][:].tolist())

source_descriptions = source_data['descriptions'][:].tolist()
target_descriptions = []
print(source_descriptions[0])
elements = [0, 1, 2]

for desc in source_descriptions:
    for elem in elements:
        array = remove_element(desc, elem)

    target_descriptions.append(array)

target_data.create_dataset("descriptions", data=target_descriptions)