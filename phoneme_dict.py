import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import ko_audio_dataset_utils as kadu
from functools import reduce
import os

class PhonemeDict:
  def __init__ (self):
    self.BLANK_INDEX = 0
    self.START_INDEX = 1
    self.END_INDEX = 2
    self.SPACE_INDEX = 3
    self.OOV_INDEX = 4

class PhonemeDictGenerator (PhonemeDict):
  def __init__ (self, labels):
    super(PhonemeDictGenerator, self).__init__()
    self.phonemes = ['<BLANK>', '<START>', '<END>', ' ']

    labels = list(filter(lambda label: not kadu.check_skip_label(label), labels))
    texts = []
    for label in labels:
      for phoneme in label:
        if not kadu.isHangeul(phoneme) and not phoneme == ' ' and not phoneme == '?':
          break
        for pure in kadu.pureosseugi(phoneme):
          texts.append(pure)

    tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)

    for i in range(1, len(tokenizer.index_word) + 1):
      self.phonemes.append(tokenizer.index_word[i])

    self.phoneme_index = {}
    for i, phoneme in enumerate(self.phonemes):
      self.phoneme_index[phoneme] = i
    
    self.VOCAB_SIZE = len(self.phonemes)
  
  def save (self, filename):
    with open(filename, 'wb') as f:
      pickle.dump((self.phonemes, self.phoneme_index), f)


class PhonemeDictLoader (PhonemeDict):
  def __init__ (self, filename):
    super(PhonemeDictLoader, self).__init__()
    with open(filename, 'rb') as f:
      self.phonemes, self.phoneme_index = pickle.load(f)

    self.VOCAB_SIZE = len(self.phonemes)

if __name__ == '__main__':
  with open(os.path.join('..', 'stt/ko-audio-dataset', 'sentence-script/KsponSpeech_scripts/train.trn'), 'r', encoding='utf-8') as f:
    lines = f.readlines()
  labels = list(map(lambda line: kadu.sentence_filter(line.split(' :: ')[1]), lines))
  labels = list(filter(lambda label: not kadu.check_skip_label(label), labels))
  phoneme_dict = PhonemeDictGenerator(labels)
  phoneme_dict.save('saves/phoneme_dict.pickle')
