import h5py
import random
import os
import time

class DatasetShuffler:
  def __init__ (self, original_dataset_path, chunk_size):
    self.original_dataset = h5py.File(original_dataset_path, 'r')
    self.chunk_size = chunk_size
  
  def shuffle (self, path):
    if os.path.exists(path):
      os.remove(path)
    f = h5py.File(path, 'w')
    f.create_dataset('mel_specs', dtype=h5py.vlen_dtype('float32'), shape=(self.original_dataset['mel_specs'].len(), ), chunks=(self.chunk_size, ))
    f.create_dataset('mel_specs_shape', dtype='int32', shape=(self.original_dataset['mel_specs_shape'].len(), 2), chunks=(self.chunk_size, 2))
    f.create_dataset('labels', dtype=h5py.vlen_dtype('int32'), shape=(self.original_dataset['labels'].len(), ), chunks=(self.chunk_size, ))

    idx_map = list(range(self.original_dataset['mel_specs'].len()))
    random.shuffle(idx_map)

    for i, j in zip(range(len(idx_map)), idx_map):
      f['mel_specs'][i] = self.original_dataset['mel_specs'][j]
      f['mel_specs_shape'][i] = self.original_dataset['mel_specs_shape'][j]
      f['labels'][i] = self.original_dataset['labels'][j]

if __name__ == '__main__':
  shuffler = DatasetShuffler('saves/no_chunk_dataset.h5', 'saves/cur_dataset.h5', 20000)
  start = time.time()
  shuffler.shuffle()
  end = time.time()
  print(end - start)
