import tensorflow as tf
from model import Transducer
from phoneme_dict import PhonemeDictLoader
import h5py
import numpy as np
import ko_audio_dataset_utils as kadu

class AudioDecoder:
  def __init__ (self, transducer):
    self.transducer = transducer
  
  def decode (self, spectrogram):
    label = [0]
    for i in range(spectrogram.shape[0]):
      sub_spectrogram = spectrogram[0: i + 1]
      sub_spectrogram = tf.convert_to_tensor(sub_spectrogram)
      sub_spectrogram = tf.expand_dims(sub_spectrogram, axis=0)
      sub_label = tf.constant([label], dtype=tf.int32)
      sub_rst = self.transducer((sub_spectrogram, sub_label))
      sub_rst = tf.argmax(sub_rst[0][-1][-1], axis=-1).numpy()
      if sub_rst != 0:
        label.append(sub_rst)
    return label

def decode_with_hint (logits):
  rst = []
  x, y = 0, 0
  while x < logits.shape[2] and y < logits.shape[1]:
    sel = tf.argmax(logits[0][y][x], axis=-1).numpy()
    rst.append(sel)
    if sel == 0:
      y += 1
    else:
      x += 1
  return list(filter(lambda x: x != 0, rst))

if __name__ == '__main__':
  EMBEDDING_DIM = 128
  UNITS = 1024
  CODER_OUTPUT_DIM = 320
  JOINT_NET_INNER_DIM = 320

  phoneme_dict = PhonemeDictLoader('saves/phoneme_dict.pickle')
  transducer = Transducer(EMBEDDING_DIM, UNITS, CODER_OUTPUT_DIM, JOINT_NET_INNER_DIM, len(phoneme_dict.phonemes))
  transducer.load_weights('saves/train_state5/checkpoints/ep0_no216032/ckpt')
  decoder = AudioDecoder(transducer)
  f = h5py.File('saves/dataset_large.h5')
  mel_specs = [np.reshape(mel_spec, mel_spec_shape) for mel_spec, mel_spec_shape in zip(f['mel_specs'][0: 10], f['mel_specs_shape'][0: 10])]
  for mel_spec, label in zip(mel_specs, f['labels'][0: 10]):
    print(kadu.moasseugi(kadu.vector_to_sentence(phoneme_dict, label)))
    print(
      kadu.moasseugi(
        kadu.vector_to_sentence(
          phoneme_dict,
          decode_with_hint(
            transducer((
              tf.expand_dims(tf.constant(mel_spec), axis=0),
              tf.expand_dims(tf.concat([tf.constant([0], dtype=tf.int32), tf.constant(label)], axis=0), axis=0)
            ))
          )
        )
      )
    )
    print(kadu.moasseugi(kadu.vector_to_sentence(phoneme_dict, decoder.decode(mel_spec)[1: ])))
    print()
