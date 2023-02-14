import h5py
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import ko_audio_dataset_utils as kadu
from phoneme_dict import PhonemeDictLoader

f = h5py.File('saves/dataset.h5')
mel_spec = np.reshape(f['mel_specs'][0], f['mel_specs_shape'][0])
label = f['labels'][0]
phoneme_dict = PhonemeDictLoader('saves/phoneme_dict.pickle')
original_sentence = kadu.moasseugi(kadu.vector_to_sentence(phoneme_dict, label))
print(original_sentence)

plt.subplot(2, 1, 1)
librosa.display.specshow(mel_spec.transpose(), y_axis='log', x_axis='time')
plt.colorbar()
plt.tight_layout()
print(mel_spec.transpose())

mel_spec2 = kadu.extract_spectrogram(
  '../stt/ko-audio-dataset/audio/KsponSpeech_04/KsponSpeech_0469/KsponSpeech_468805.pcm',
  16000, 512, 256, 128
)
plt.subplot(2, 1, 2)
librosa.display.specshow(mel_spec2.transpose(), y_axis='log', x_axis='time')
plt.colorbar()
plt.tight_layout()
print(mel_spec2.transpose())

plt.show()
