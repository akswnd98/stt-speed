import h5py
from phoneme_dict import PhonemeDictLoader
import ko_audio_dataset_utils as kadu

dataset = h5py.File('saves/dataset.h5')
phoneme_dict = PhonemeDictLoader('saves/phoneme_dict.pickle')

def vector_to_text (vector, phonemes):
  return kadu.moasseugi([phonemes[v] for v in vector])

print(vector_to_text(dataset['labels'][0], phoneme_dict.phonemes))
print(vector_to_text(dataset['labels'][612], phoneme_dict.phonemes))
print(vector_to_text(dataset['labels'][6129], phoneme_dict.phonemes))
print(vector_to_text(dataset['labels'][312915], phoneme_dict.phonemes))
