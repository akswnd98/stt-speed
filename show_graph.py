import matplotlib.pyplot as plt
import h5py
import numpy as np

f = h5py.File('saves/train_state8/loss.h5', 'r')
plt.plot(np.arange(0, f['loss'][10: ].shape[0] * 32, 32), list(f['loss'][10: ]))

plt.show()
