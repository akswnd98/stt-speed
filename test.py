from trainer import InitialLossSaver
import h5py
import pickle

with open('saves/train_state7/prev_save_time.pickle', 'rb') as f:
  a = pickle.load(f)

print(a)
