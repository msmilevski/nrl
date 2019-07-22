import h5py
import pickle

pairs = []
for i in range(100):
    d = h5py.File('dataset/resnet152/image_embeddings_' + str(i) + '.hdf5', 'r')
    temp_id = d['image_id'][()]

    for i, id in enumerate(temp_id):
        pairs.append((id, i))


d = dict(pairs)
pickle.dump(d, open('dataset/img_id_map.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)