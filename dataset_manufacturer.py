import ko_audio_dataset_utils as kadu
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from phoneme_dict import PhonemeDictLoader
import matplotlib.pyplot as plt
import librosa
import librosa.display
import h5py
import multiprocessing as mp
import random
import numpy as np
import time

class DatasetManufacturer:
  def __init__ (self, phoneme_dict, workers_num, work_size, chunk_size, mel_spec_size_limit, label_size_limit):
    self.phoneme_dict = phoneme_dict
    self.workers_num = workers_num
    self.chunk_size = chunk_size
    self.work_size = work_size
    self.mel_spec_size_limit = mel_spec_size_limit
    self.label_size_limit = label_size_limit

  def manufacture (self, path, output_path):
    with open(os.path.join(path, 'sentence-script/KsponSpeech_scripts/train.trn'), 'r', encoding='utf-8') as f:
      lines = f.readlines()
    random.shuffle(lines)
    f = h5py.File(output_path, 'w')

    f.create_dataset('mel_specs', dtype=h5py.vlen_dtype('float32'), maxshape=(len(lines), ), shape=(0, ), chunks=(1, ))
    f.create_dataset('mel_specs_shape', dtype='int32', maxshape=(len(lines), 2), shape=(0, 2), chunks=(1, 2))
    f.create_dataset('labels', dtype=h5py.vlen_dtype('int32'), maxshape=(len(lines), ), shape=(0, ), chunks=(1, ))
    cur_idx = 0
    for i in range(0, len(lines), self.workers_num * self.work_size):
      print(i)
      works = []
      pool = mp.Pool(self.workers_num)
      for j in range(self.workers_num):
        works.append(lines[min(len(lines), i + j * self.work_size): min(len(lines), i + (j + 1) * self.work_size)])
      sub_results = pool.starmap(
        process,
        list(zip(
          works,
          [path] * self.workers_num,
          [self.phoneme_dict.phoneme_index] * self.workers_num,
          [self.mel_spec_size_limit] * self.workers_num,
          [self.label_size_limit] * self.workers_num
        ))
      )
      pool.close()
      pool.join()
      mel_specs = []
      labels = []
      for sub_result in sub_results:
        for mel_spec in sub_result[0]:
          mel_specs.append(mel_spec)
        for label in sub_result[1]:
          labels.append(label)
      f['mel_specs'].resize((cur_idx + len(mel_specs), ))
      f['mel_specs'][cur_idx: cur_idx + len(mel_specs)] = [np.reshape(mel_spec, -1) for mel_spec in mel_specs]
      f['mel_specs_shape'].resize((cur_idx + len(mel_specs), 2))
      f['mel_specs_shape'][cur_idx: cur_idx + len(mel_specs)] = [(len(mel_spec), len(mel_spec[0])) for mel_spec in mel_specs]
      f['labels'].resize((cur_idx + len(labels), ))
      f['labels'][cur_idx: cur_idx + len(labels)] = labels
      cur_idx += len(mel_specs)
    f.close()

def process (lines, path, phoneme_index, mel_spec_size_limit, label_size_limit, sr=16000, n_fft=512, hop_len=256, n_mels=128):
  def process_line (line: str):
    line = line.split(' :: ')
    line[0] = os.path.join(path, 'audio', line[0])
    line[1] = kadu.sentence_filter(line[1])
    return line
  data_pairs = list(map(process_line, lines))
  data_pairs = list(filter(lambda data_pair: not kadu.check_skip_label(data_pair[1]), data_pairs))

  def process_pair (pair):
    return (
      kadu.extract_spectrogram(pair[0], sr=sr, n_fft=n_fft, hop_len=hop_len, n_mels=n_mels),
      [phoneme_index[phoneme] for phoneme in kadu.pureosseugi(pair[1])]
    )
  data_pairs = list(map(process_pair, data_pairs))
  data_pairs = list(filter(lambda data_pair: data_pair[0].shape[0] <= mel_spec_size_limit and len(data_pair[1]) < label_size_limit, data_pairs))
  mel_specs = [mel_spec for mel_spec, label in data_pairs]
  labels = [label for mel_spec, label in data_pairs]
  return (mel_specs, labels)

if __name__ == '__main__':
  start = time.time()
  phoneme_dict = PhonemeDictLoader('saves/phoneme_dict.pickle')
  manaufacturer = DatasetManufacturer(phoneme_dict, 14, 2000, 20000, 160, 50)
  manaufacturer.manufacture('../stt/ko-audio-dataset', 'saves/no_chunk_dataset2.h5')
  end = time.time()
  print(end - start)