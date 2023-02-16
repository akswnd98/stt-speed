import h5py

with h5py.File('saves/train-state5/loss.h5', 'r') as f:
  print(f['loss'][0])
  f.close()