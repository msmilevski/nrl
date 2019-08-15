import h5py

for i in range(99):
    name = '/disk/scratch_big/1885778/image_features_' + str(i) + '.hdf5'
    try:
        f = h5py.File(name, 'r')
    except:
        print(name)