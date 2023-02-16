from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import numpy as np
import librosa
from functools import reduce
import os
import pickle
from model import Transducer
import tensorflow as tf
from rnnt_loss import rnnt_loss
from dataset_shuffler import DatasetShuffler
import h5py
import tensorflow as tf
from phoneme_dict import PhonemeDictLoader

class Saver:
  def save (self, path, cur_epoch, cnt):
    pass

class InitialSaver:
  def initial_save (self):
    pass

class TrainPolicy:
  def train (self):
    pass

class StepPolicy (TrainPolicy):
  def __init__ (
    self,
    original_dataset_path: str,
    dataset_path: str,
    dataset_shuffler: any,
    optimizer: tf.keras.optimizers.Adam,
    epochs: int,
    batch_size: int,
    batch_num: int,
    step_performer: any,
    step_end_handlers: any,
    epoch_end_handlers: any,
    cur_epoch: int,
    cnt: int
  ):
    self.original_dataset = h5py.File(original_dataset_path, 'r')
    self.dataset_path = dataset_path
    self.optimizer = optimizer
    self.epochs = epochs
    self.batch_size = batch_size
    self.batch_num = batch_num
    self.step_size = self.batch_size * self.batch_num

    self.step_performer = step_performer
    self.step_end_handlers = step_end_handlers
    self.epoch_end_handlers = epoch_end_handlers
    self.dataset_shuffler = dataset_shuffler

    self.cur_epoch = cur_epoch
    self.cnt = cnt

  def train (self):
    for i in range(self.cur_epoch, self.epochs):
      dataset = h5py.File(self.dataset_path, 'r')
      for j in range(self.cnt, self.step_size * (dataset['mel_specs'].len() // self.step_size), self.step_size):
        loss = self.step_performer.step(
          self.optimizer,
          self.batch_size,
          self.batch_num,
          [
            np.reshape(dataset['mel_specs'][k], dataset['mel_specs_shape'][k])
            for k in range(j, j + self.step_size)
          ],
          [np.concatenate([np.array([0], dtype=np.int32), label]) for label in dataset['labels'][j: j + self.step_size]]
        )
        self.cnt += self.step_size
        for step_end_handler in self.step_end_handlers:
          step_end_handler.handle(self.cur_epoch, self.cnt, loss, self.step_size)

      self.cur_epoch += 1
      self.cnt = 0
      dataset.close()
      self.dataset_shuffler.shuffle(self.dataset_path)
      for epoch_end_handler in self.epoch_end_handlers:
        epoch_end_handler.handle(self.epochs, self.cur_epoch, self.cnt)

class SavableStepPolicy (StepPolicy, Saver, InitialSaver):
  def __init__ (
    self,
    save_path,
    original_dataset_path,
    dataset_path,
    dataset_shuffler,
    savable_save_notifier,
    optimizer,
    epochs,
    batch_size,
    batch_num,
    step_performer,
    step_end_handlers,
    epoch_end_handlers,
    cur_epoch,
    cnt,
    prev_save_time
  ):
    StepPolicy.__init__(
      self,
      original_dataset_path,
      dataset_path,
      dataset_shuffler,
      optimizer,
      epochs,
      batch_size,
      batch_num,
      step_performer,
      step_end_handlers,
      epoch_end_handlers,
      cur_epoch,
      cnt
    )
    savable_save_notifier.savers.append(self)
    savable_save_notifier.savers.append(self.step_performer)
    self.step_end_handlers.append(
      savable_save_notifier
    )
    self.epoch_end_handlers.append(
      SavableEpochSaveNotifier(
        save_path,
        [
          self,
          self.step_performer,
        ],
        prev_save_time
      )
    )
  
  def save (self, path, cur_epoch, cnt):
    with open(os.path.join(path, 'progress.pickle'), 'wb') as f:
      pickle.dump((self.cur_epoch, self.cnt), f)

  def initial_save (self, path):
    with open(os.path.join(path, 'progress.pickle'), 'wb') as f:
      pickle.dump((self.cur_epoch, self.cnt), f)
    with open(os.path.join(path, 'policy_meta.pickle'), 'wb') as f:
      pickle.dump((self.batch_num, self.batch_size), f)

class TrainStepPerformer:
  def __init__ (self, transducer):
    self.transducer = transducer

  def step (self, optimizer, batch_size, batch_num, spectrograms, labels):
    acc_grads = [tf.zeros_like(var) for var in self.transducer.trainable_variables]
    acc_loss = 0
    for i in range(0, len(labels), batch_size):
      batch_labels = labels[i: i + batch_size]
      batch_spectrograms = spectrograms[i: i + batch_size]
      label_lengths = self.get_label_lengths(batch_labels)
      label_lengths = tf.convert_to_tensor(label_lengths, dtype=tf.int32)
      batch_labels = pad_sequences(batch_labels, maxlen=self.get_labels_maxlen(batch_labels), truncating='post', padding='post')
      batch_labels = tf.constant(batch_labels, dtype=tf.int32)
      spectrogram_lengths = self.get_spectrogram_lengths(batch_spectrograms)
      spectrogram_lengths = tf.convert_to_tensor(spectrogram_lengths, dtype=tf.int32)
      batch_spectrograms = tf.cast(pad_sequences(batch_spectrograms, maxlen=self.get_spectrograms_maxlen(batch_spectrograms), truncating='post', padding='post'), dtype=tf.float32)
      with tf.GradientTape() as tape:
        logits = self.transducer([batch_spectrograms, batch_labels])
        loss = rnnt_loss(
          logits,
          batch_labels,
          spectrogram_lengths,
          label_lengths
        )
      acc_loss += loss.numpy()
      grads = tape.gradient(loss, self.transducer.trainable_variables)
      acc_grads = [acc_grad + grad for acc_grad, grad in zip(acc_grads, grads)]
    acc_grads = [acc_grad / (batch_size * batch_num) for acc_grad in acc_grads]
    optimizer.apply_gradients(zip(acc_grads, self.transducer.trainable_variables))
    
    return acc_loss / (batch_size * batch_num)
  
  def get_labels_maxlen (self, labels):
    ret = 0
    for label in labels:
      ret = max(ret, label.shape[0])
    return ret
  
  def get_label_lengths (self, labels):
    ret = list(map(lambda label: label.shape[0], labels))
    return ret
  
  def get_spectrograms_maxlen (self, spectrograms):
    ret = 0
    for spectrogram in spectrograms:
      ret = max(ret, spectrogram.shape[0])
    return ret

  def get_spectrogram_lengths (self, spectrograms):
    ret = list(map(lambda x: x.shape[0], spectrograms))
    return ret

class SavableTrainStepPerformer (TrainStepPerformer, Saver, InitialSaver):
  def __init__ (self, transducer, embedding_dim, units, coder_output_dim, joint_net_inner_dim, vocab_size):
    TrainStepPerformer.__init__(self, transducer)
    self.embedding_dim = embedding_dim
    self.units = units
    self.coder_output_dim = coder_output_dim
    self.joint_net_inner_dim = joint_net_inner_dim
    self.vocab_size = vocab_size

  def save (self, path, cur_epoch, cnt):
    self.transducer.save_weights(os.path.join(path, 'checkpoints/ep{}_no{}/ckpt'.format(cur_epoch, cnt)))

  def initial_save (self, path):
    self.transducer.save_weights(os.path.join(path, 'checkpoints/ep{}_no{}/ckpt'.format(0, 0)))
    with open(os.path.join(path, 'model_meta.pickle'), 'wb') as f:
      pickle.dump((
        self.embedding_dim,
        self.units,
        self.coder_output_dim,
        self.joint_net_inner_dim,
        self.vocab_size
      ), f)

class SavableTrainStepPerformerLoader:
  @staticmethod
  def load (path):
    with open(os.path.join(path, 'model_meta.pickle'), 'rb') as f:
      model_meta = pickle.load(f)
    with open(os.path.join(path, 'progress.pickle'), 'rb') as f:
      progress = pickle.load(f)
    transducer = Transducer(*model_meta)
    transducer([
      tf.keras.Input(shape=(None, 80), dtype=tf.float32),
      tf.keras.Input(shape=(None, ), dtype=tf.int32)
    ])
    transducer.load_weights(os.path.join(path, 'checkpoints/ep{}_no{}/ckpt'.format(progress[0], progress[1])))
    return SavableTrainStepPerformer(
      transducer,
      *model_meta
    )

class PrevSaveTime:
  def __init__ (self, prev_save_time):
    self.prev_save_time = prev_save_time
  
  def update (self, prev_save_time):
    self.prev_save_time = prev_save_time

class EpochEndHandler:
  def handle (self, epochs, cur_epoch, cnt):
    pass

class EpochPrinter (EpochEndHandler):
  def handle (self, epochs, cur_epoch, cnt):
    print('epoch: {} / {}'.format(cur_epoch, epochs))

class EpochSaveNotifier (EpochEndHandler):
  def __init__ (self, path, savers):
    super(EpochSaveNotifier, self).__init__()
    self.path = path
    self.savers = savers

  def handle (self, epochs, cur_epoch, cnt):
    for saver in self.savers:
      saver.save(self.path, cur_epoch, cnt)
  
class SavableEpochSaveNotifier (EpochSaveNotifier):
  def __init__ (self, path, savers, prev_save_time):
    super(SavableEpochSaveNotifier, self).__init__(path, savers)
    self.prev_save_time = prev_save_time
    self.savers.append(self)
  
  def save (self, path, cur_epoch, cnt):
    with open(os.path.join(path, 'prev_save_time.pickle'), 'wb') as f:
      pickle.dump(0, f)
    self.prev_save_time.update(0)

class StepEndHandler:
  def handle (self, cur_epoch, cnt, loss, step_size):
    pass

class StepPrinter (StepEndHandler):
  def handle (self, cur_epoch, cnt, loss, step_size):
    print('epoch, cnt: {}, {}'.format(cur_epoch, cnt))
    print('loss: {}'.format(loss / step_size))

class SaveNotifier (StepEndHandler):
  def __init__ (self, path, savers, save_time_delta, prev_save_time):
    super(SaveNotifier, self).__init__()
    self.path = path
    self.savers = savers
    self.save_time_delta = save_time_delta
    self.prev_save_time = prev_save_time

  def handle (self, cur_epoch, cnt, loss, step_size):
    if self.prev_save_time.prev_save_time + self.save_time_delta <= cnt:
      for saver in self.savers:
        saver.save(self.path, cur_epoch, cnt)
      self.prev_save_time.update(cnt)

class SavableSaveNotifier (SaveNotifier, Saver, InitialSaver):
  def __init__ (self, path, savers, save_time_delta, prev_save_time):
    super(SavableSaveNotifier, self).__init__(path, savers, save_time_delta, prev_save_time)
    self.savers.append(self)
  
  def save (self, path, cur_epoch, cnt):
    with open(os.path.join(path, 'prev_save_time.pickle'), 'wb') as f:
      pickle.dump(cnt, f)
  
  def initial_save (self, path):
    with open(os.path.join(path, 'prev_save_time.pickle'), 'wb') as f:
      pickle.dump(0, f)

class InitialSaveNotifier:
  def __init__ (self, path, initial_savers):
    self.path = path
    self.initial_savers = initial_savers

  def initial_save (self):
    os.makedirs(self.path, exist_ok=True)
    for initial_saver in self.initial_savers:
      initial_saver.initial_save(self.path)

class InitialSavableDatasetShuffler (DatasetShuffler):
  def initial_save (self, path):
    self.shuffle(os.path.join(path, 'cur_dataset.h5'))

class SimplePolicyGenerator:
  @staticmethod
  def generate (
    save_path, original_dataset_path,
    embedding_dim, units, coder_output_dim, joint_net_inner_dim, vocab_size,
    epoch, batch_size, batch_num, learning_rate, save_time_delta
  ):
    prev_save_time = PrevSaveTime(0)
    savable_save_notifier = SavableSaveNotifier(save_path, [], save_time_delta, prev_save_time)
    step_size = batch_size * batch_num
    cur_epoch = 0
    cnt = 0
    model_meta = (embedding_dim, units, coder_output_dim, joint_net_inner_dim, vocab_size)
    transducer = Transducer(*model_meta)
    transducer([
      tf.keras.Input(shape=(None, 80), dtype=tf.float32),
      tf.keras.Input(shape=(None, ), dtype=tf.int32)
    ])

    dataset_path = os.path.join(save_path, 'cur_dataset.h5')
    policy = SavableStepPolicy(
      save_path,
      original_dataset_path,
      dataset_path,
      InitialSavableDatasetShuffler(original_dataset_path, 20000),
      savable_save_notifier,
      tf.keras.optimizers.Adam(learning_rate=learning_rate),
      epoch,
      batch_size,
      batch_num,
      SavableTrainStepPerformer(
        transducer,
        *model_meta
      ),
      [StepPrinter()],
      [EpochPrinter()],
      cur_epoch,
      cnt,
      prev_save_time
    )
    return \
      policy, \
      InitialSaveNotifier(
        save_path, [
          policy,
          policy.step_performer,
          savable_save_notifier,
          InitialSavableDatasetShuffler(original_dataset_path, 20000)
        ]
      )

class SimplePolicyLoader:
  @staticmethod
  def load (save_path, original_dataset_path, epoch, batch_size, batch_num, learning_rate, save_time_delta):
    with open(os.path.join(save_path, 'progress.pickle'), 'rb') as f:
      progress = pickle.load(f)
    prev_save_time = PrevSaveTime(
      pickle.load(
        open(os.path.join(save_path, 'prev_save_time.pickle'), 'rb')
      )
    )
    dataset_path = os.path.join(save_path, 'cur_dataset.h5')
    policy = SavableStepPolicy(
      save_path,
      original_dataset_path,
      os.path.join(save_path, 'cur_dataset.h5'),
      InitialSavableDatasetShuffler(original_dataset_path, 20000),
      SavableSaveNotifier(
        save_path,
        [],
        save_time_delta,
        prev_save_time
      ),
      tf.keras.optimizers.Adam(learning_rate=learning_rate),
      epoch,
      batch_size,
      batch_num,
      SavableTrainStepPerformerLoader.load(save_path),
      [StepPrinter()],
      [EpochPrinter()],
      progress[0],
      progress[1],
      prev_save_time
    )
    return policy
  
if __name__ == '__main__':
  EMBEDDING_DIM = 128
  UNITS = 1024
  CODER_OUTPUT_DIM = 320
  JOINT_NET_INNER_DIM = 320

  EPOCHS = 20
  BATCH_SIZE = 8
  BATCH_NUM = 4
  LEARNING_RATE = 0.00001
  SAVE_PATH = 'saves/train_state4'
  SAVE_TIME_DELTA = 10000

  # policy, initial_save_notifier = SimplePolicyGenerator.generate(
  #   SAVE_PATH,
  #   'saves/dataset_large.h5',
  #   EMBEDDING_DIM,
  #   UNITS,
  #   CODER_OUTPUT_DIM,
  #   JOINT_NET_INNER_DIM,
  #   len(PhonemeDictLoader('saves/phoneme_dict.pickle').phonemes),
  #   EPOCHS,
  #   BATCH_SIZE,
  #   BATCH_NUM,
  #   LEARNING_RATE,
  #   SAVE_TIME_DELTA
  # )
  # policy.step_performer.transducer.summary()
  # initial_save_notifier.initial_save()

  policy = SimplePolicyLoader.load(
    SAVE_PATH,
    'saves/dataset_large.h5',
    EPOCHS,
    BATCH_SIZE,
    BATCH_NUM,
    LEARNING_RATE,
    SAVE_TIME_DELTA
  )

  with tf.device('/gpu:0'):
    policy.train()

  # 그리고 학습 결과도 저장할 것.
