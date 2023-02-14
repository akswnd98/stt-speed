import h5py
import numpy as np

h = h5py.File('hello.h5', 'w')
a = h.create_group('a')
dataset = a.create_dataset('dataset', maxshape=(100000, 10, 10), dtype='int')
dataset = np.zeros((1, 10, 10))
# dataset.resize((100, 10, 10))
# dataset[1: ] = np.zeros((99, 10, 10))

print(np.array(dataset))

# print(np.array(a['b']).shape)

