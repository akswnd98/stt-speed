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

# initial save

class SimpleInitialSaver:
  @staticmethod
  def initial_save (
    base_path,
    original_dataset_path,
    chunk_size,
    model_meta,
    loss_maxshape,
    loss_chunks
  ):
    InitialSaveNotifier(base_path, [
      InitialPickleSaver(base_path, 'model_meta.pickle', model_meta),
      InitialPickleSaver(base_path, 'prev_save_time.pickle', 0),
      InitialPickleSaver(base_path, 'cur_epoch.pickle', 0),
      InitialPickleSaver(base_path, 'cnt.pickle', 0),
      InitialLossSaver(base_path, loss_maxshape, loss_chunks),
      InitialTransducerSaver(base_path, Transducer(*model_meta)),
      InitialCurDatasetSaver(base_path, original_dataset_path, chunk_size)
    ]).initial_save()

# simple generator

class SimpleGenerator:
  @staticmethod
  def generate (
    learning_rate,
    epochs,
    step_size,
    batch_size,
    batch_num,
    base_path,
    original_dataset_path,
    chunk_size,
    save_time_delta,
    model_meta,
    n_mels
  ):
    cur_epoch_model = CurEpochModel(0, [])
    cnt_model = CntModel(0, [])
    prev_save_time_model = PrevSaveTimeModel(0, [])
    loss_model = LossModel(0, [])
    loss_saver = LossSaver(base_path, loss_model)
    loss_model.register(LossObserver(loss_saver))
    transducer = Transducer(*model_meta)
    transducer([
      tf.keras.Input(shape=(None, n_mels), dtype=tf.float32),
      tf.keras.Input(shape=(None, ), dtype=tf.int32)
    ])
    return TrainerInst(
      EndHandlableEpochRunner(
        EpochRunnerInst(
          EndHandlableStepRunner(
            StepRunnerInst(transducer, cnt_model, loss_model, step_size),
            [
              ProgressionPrinter(cur_epoch_model, cnt_model, loss_model),
              StepEndProgressionSaveNotifier([
                CntSaver(base_path, cnt_model),
                PrevSaveTimeSaver(base_path, prev_save_time_model),
                loss_saver,
                TransducerSaver(base_path, transducer, cur_epoch_model, cnt_model)
              ], save_time_delta, prev_save_time_model, cnt_model)
            ]
          ),
          base_path,
          tf.keras.optimizers.Adam(learning_rate=learning_rate),
          step_size,
          batch_size,
          batch_num,
          cur_epoch_model,
          cnt_model
        ),
        [
          ProgressionPrinter(cur_epoch_model, cnt_model, loss_model),
          EpochEndProgressionSaveNotifier([
            CurEpochSaver(base_path, cur_epoch_model),
            CntSaver(base_path, cnt_model),
            PrevSaveTimeSaver(base_path, prev_save_time_model),
            loss_saver,
            TransducerSaver(base_path, transducer, cur_epoch_model, cnt_model)
          ], prev_save_time_model, cnt_model)
        ]
      ),
      cur_epoch_model,
      cnt_model,
      epochs,
      DatasetShuffler(original_dataset_path, chunk_size),
      base_path
    )

# simple loader

class SimpleLoader:
  @staticmethod
  def load (
    base_path,
    learning_rate,
    epochs,
    step_size,
    batch_size,
    batch_num,
    original_dataset_path,
    chunk_size,
    save_time_delta,
    n_mels
  ):
    cur_epoch_model = CurEpochModelLoader.load(base_path)
    cnt_model = CntModelLoader.load(base_path)
    prev_save_time_model = PrevSaveTimeModelLoader.load(base_path)
    loss_model = LossModel(0, [])
    loss_saver = LossSaver(base_path, loss_model)
    loss_model.register(LossObserver(loss_saver))
    model_meta = ModelMetaLoader.load(base_path)
    transducer = Transducer(*model_meta)
    transducer.load_weights(os.path.join(base_path, 'checkpoints/ep{}_no{}/ckpt'.format(cur_epoch_model.get_value(), cnt_model.get_value())))
    transducer([
      tf.keras.Input(shape=(None, n_mels), dtype=tf.float32),
      tf.keras.Input(shape=(None, ), dtype=tf.int32)
    ])
    return TrainerInst(
      EndHandlableEpochRunner(
        EpochRunnerInst(
          EndHandlableStepRunner(
            StepRunnerInst(transducer, cnt_model, loss_model, step_size),
            [
              ProgressionPrinter(cur_epoch_model, cnt_model, loss_model),
              StepEndProgressionSaveNotifier([
                CntSaver(base_path, cnt_model),
                PrevSaveTimeSaver(base_path, prev_save_time_model),
                loss_saver,
                TransducerSaver(base_path, transducer, cur_epoch_model, cnt_model)
              ], save_time_delta, prev_save_time_model, cnt_model)
            ]
          ),
          base_path,
          tf.keras.optimizers.Adam(learning_rate=learning_rate),
          step_size,
          batch_size,
          batch_num,
          cur_epoch_model,
          cnt_model
        ),
        [
          ProgressionPrinter(cur_epoch_model, cnt_model, loss_model),
          EpochEndProgressionSaveNotifier([
            CurEpochSaver(base_path, cur_epoch_model),
            CntSaver(base_path, cnt_model),
            PrevSaveTimeSaver(base_path, prev_save_time_model),
            loss_saver,
            TransducerSaver(base_path, transducer, cur_epoch_model, cnt_model)
          ], prev_save_time_model, cnt_model)
        ]
      ),
      cur_epoch_model,
      cnt_model,
      epochs,
      DatasetShuffler(original_dataset_path, chunk_size),
      base_path
    )

# trainer

class Trainer:
  def train (self):
    pass

class TrainerInst(Trainer):
  def __init__ (self, epoch_runner, cur_epoch_model, cnt_model, epochs, dataset_shuffler, base_path):
    super(TrainerInst, self).__init__()
    self.epoch_runner = epoch_runner
    self.cur_epoch_model = cur_epoch_model
    self.cnt_model = cnt_model
    self.epochs = epochs
    self.dataset_shuffler = dataset_shuffler
    self.base_path = base_path

  def train (self):
    for _ in range(self.cur_epoch_model.get_value(), self.epochs):
      self.epoch_runner.run_epoch()
      self.dataset_shuffler.shuffle(os.path.join(self.base_path, 'cur_dataset.h5'))

class EndHandlableTrainer (Trainer):
  def __init__ (self, trainer, handlers):
    super(EndHandlableTrainer, self).__init__()
    self.trainer = trainer
    self.handlers = handlers
  
  def train (self):
    self.trainer.train()
    for handler in self.handlers:
      handler.handle()

# epoch runner

class EpochRunner:
  def run_epoch (self):
    pass

class EpochRunnerInst (EpochRunner):
  def __init__ (self, step_runner, base_path, optimizer, step_size, batch_size, batch_num, cur_epoch_model, cnt_model):
    super(EpochRunnerInst, self).__init__()
    self.step_runner = step_runner
    self.dataset_path = os.path.join(base_path, 'cur_dataset.h5')
    self.optimizer = optimizer
    self.step_size = step_size
    self.batch_size = batch_size
    self.batch_num = batch_num
    self.cur_epoch_model = cur_epoch_model
    self.cnt_model = cnt_model

  def run_epoch (self):
    dataset = h5py.File(self.dataset_path, 'r')
    while self.cnt_model.get_value() + self.step_size <= dataset['mel_specs'].len():
      self.step_runner.step(
        self.optimizer,
        self.batch_size,
        self.batch_num,
        [
          np.reshape(mel_spec, mel_spec_shape)
          for mel_spec, mel_spec_shape in zip(
            dataset['mel_specs'][self.cnt_model.get_value(): self.cnt_model.get_value() + self.step_size],
            dataset['mel_specs_shape'][self.cnt_model.get_value(): self.cnt_model.get_value() + self.step_size]
          )
        ],
        dataset['labels'][self.cnt_model.get_value(): self.cnt_model.get_value() + self.step_size]
      )
    self.cur_epoch_model.notify(self.cur_epoch_model.get_value() + 1)
    self.cnt_model.notify(0)

class EndHandlableEpochRunner (EpochRunner):
  def __init__ (self, epoch_runner, handlers):
    super(EndHandlableEpochRunner, self).__init__()
    self.epoch_runner = epoch_runner
    self.handlers = handlers
  
  def run_epoch (self):
    self.epoch_runner.run_epoch()
    for handler in self.handlers:
      handler.handle()

# step runner

class StepRunner:
  def step (self, optimizer, batch_size, batch_num, spectrograms, labels):
    pass

class StepRunnerInst(StepRunner):
  def __init__ (self, transducer, cnt_model, loss_model, step_size):
    super(StepRunnerInst, self).__init__()
    self.transducer = transducer
    self.cnt_model = cnt_model
    self.loss_model = loss_model
    self.step_size = step_size

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

    self.cnt_model.notify(self.cnt_model.get_value() + self.step_size)
    self.loss_model.notify(acc_loss / (batch_size * batch_num))
  
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

class EndHandlableStepRunner (StepRunner):
  def __init__ (self, step_runner, handlers):
    super(EndHandlableStepRunner, self).__init__()
    self.step_runner = step_runner
    self.handlers = handlers
  
  def step (self, optimizer, batch_size, batch_num, spectrograms, labels):
    self.step_runner.step(optimizer, batch_size, batch_num, spectrograms, labels)
    for handler in self.handlers:
      handler.handle()

# models

class Model:
  def get_value (self):
    pass

class Observer:
  def update (self, val):
    pass

class CurEpochModel (Model):
  def __init__ (self, cur_epoch, observers):
    self.cur_epoch = cur_epoch
    self.observers = observers

  def get_value (self):
    return self.cur_epoch

  def notify (self, cur_epoch):
    self.cur_epoch = cur_epoch
    for observer in self.observers:
      observer.update(cur_epoch)

class CntModel (Model):
  def __init__ (self, cnt, observers):
    self.cnt = cnt
    self.observers = observers

  def get_value (self):
    return self.cnt

  def notify (self, cnt):
    self.cnt = cnt
    for observer in self.observers:
      observer.update(cnt)

class PrevSaveTimeModel (Model):
  def __init__ (self, prev_save_time, observers):
    self.prev_save_time = prev_save_time
    self.observers = observers
  
  def get_value (self):
    return self.prev_save_time

  def notify (self, prev_save_time):
    self.prev_save_time = prev_save_time
    for observer in self.observers:
      observer.update(prev_save_time)

class LossModel:
  def __init__ (self, loss, observers):
    self.loss = loss
    self.observers = observers
  
  def get_value (self):
    return self.loss
  
  def notify (self, loss):
    self.loss = loss
    for observer in self.observers:
      observer.update(loss)
  
  def register (self, observer):
    self.observers.append(observer)

class LossObserver (Observer):
  def __init__ (self, loss_saver):
    super(LossObserver, self).__init__()
    self.loss_saver = loss_saver

  def update (self, loss):
    self.loss_saver.push_loss(loss)

# handlers

class EndHandler:
  def handle ():
    pass

class ProgressionPrinter (EndHandler):
  def __init__ (self, cur_epoch_model, cnt_model, loss_model):
    super(ProgressionPrinter, self).__init__()
    self.cur_epoch_model = cur_epoch_model
    self.cnt_model = cnt_model
    self.loss_model = loss_model

  def handle (self):
    print(
      'cur_epoch, cnt, loss: {}, {}, {}'.format(
        self.cur_epoch_model.get_value(),
        self.cnt_model.get_value(),
        self.loss_model.get_value()
      )
    )

class StepEndProgressionSaveNotifier (EndHandler):
  def __init__ (self, savers, save_time_delta, prev_save_time_model, cnt_model):
    super(StepEndProgressionSaveNotifier, self).__init__()
    self.savers = savers
    self.save_time_delta = save_time_delta
    self.prev_save_time_model = prev_save_time_model
    self.cnt_model = cnt_model

  def handle (self):
    if self.cnt_model.get_value() >= self.prev_save_time_model.get_value() + self.save_time_delta:
      for saver in self.savers:
        saver.save()
      self.prev_save_time_model.notify(self.cnt_model.get_value())

class EpochEndProgressionSaveNotifier (EndHandler):
  def __init__ (self, savers, prev_save_time_model, cnt_model):
    super(EpochEndProgressionSaveNotifier, self).__init__()
    self.savers = savers
    self.prev_save_time_model = prev_save_time_model
    self.cnt_model = cnt_model
  
  def handle (self):
    for saver in self.savers:
      saver.save()
    self.prev_save_time_model.notify(self.cnt_model.get_value())

# savers

class Saver:
  def save (self):
    pass

class PickleSaver (Saver):
  def __init__ (self, path, model):
    super(PickleSaver, self).__init__()
    self.path = path
    self.model = model
  
  def save (self):
    with open(self.path, 'wb') as f:
      pickle.dump(self.model.get_value(), f)

class CurEpochSaver (PickleSaver):
  def __init__ (self, base_path, cur_epoch_model):
    super(CurEpochSaver, self).__init__(os.path.join(base_path, 'cur_epoch.pickle'), cur_epoch_model)

class CntSaver (PickleSaver):
  def __init__ (self, base_path, cnt_model):
    super(CntSaver, self).__init__(os.path.join(base_path, 'cnt.pickle'), cnt_model)

class PrevSaveTimeSaver (PickleSaver):
  def __init__ (self, base_path, prev_save_time_model):
    super(PrevSaveTimeSaver, self).__init__(os.path.join(base_path, 'prev_save_time.pickle'), prev_save_time_model)

class LossSaver (Saver):
  def __init__ (self, base_path, loss_model):
    super(LossSaver, self).__init__()
    self.base_path = base_path
    self.loss_model = loss_model
    self.losses = []

  def save (self):
    f = h5py.File(os.path.join(self.base_path, 'loss.h5'), 'r+')
    start = f['loss'].len()
    end = start + len(self.losses)
    f['loss'].resize((end, ))
    f['loss'][start: end] = self.losses
    self.losses = []
    f.close()
  
  def push_loss (self, loss):
    self.losses.append(loss)

class TransducerSaver (Saver):
  def __init__ (self, base_path, transducer, cur_epoch_model, cnt_model):
    super(TransducerSaver, self).__init__()
    self.base_path = base_path
    self.transducer = transducer
    self.cur_epoch_model = cur_epoch_model
    self.cnt_model = cnt_model

  def save (self):
    self.transducer.save_weights(
      os.path.join(
        self.base_path,
        'checkpoints/ep{}_no{}/ckpt'.format(
          self.cur_epoch_model.get_value(),
          self.cnt_model.get_value()
        )
      )
    )

# loaders

class CurEpochModelLoader:
  @staticmethod
  def load (base_path):
    with open(os.path.join(base_path, 'cur_epoch.pickle'), 'rb') as f:
      cur_epoch = pickle.load(f)
    return CurEpochModel(cur_epoch, [])

class CntModelLoader:
  @staticmethod
  def load (base_path):
    with open(os.path.join(base_path, 'cnt.pickle'), 'rb') as f:
      cnt = pickle.load(f)
    return CntModel(cnt, [])
  
class PrevSaveTimeModelLoader:
  @staticmethod
  def load (base_path):
    with open(os.path.join(base_path, 'prev_save_time.pickle'), 'rb') as f:
      prev_save_time = pickle.load(f)
    return PrevSaveTimeModel(prev_save_time, [])

# initial saver

class InitialSaveNotifier:
  def __init__ (self, base_path, savers):
    self.base_path = base_path
    self.savers = savers

  def initial_save (self):
    os.makedirs(self.base_path, exist_ok=True)
    for saver in self.savers:
      saver.initial_save()

class InitialSaver:
  def __init__ (self, base_path):
    self.base_path = base_path

  def initial_save (self):
    pass

class InitialPickleSaver (InitialSaver):
  def __init__ (self, base_path, file_name, initial_value):
    super(InitialPickleSaver, self).__init__(base_path)
    self.file_name = file_name
    self.initial_value = initial_value

  def initial_save (self):
    with open(os.path.join(self.base_path, self.file_name), 'wb') as f:
      pickle.dump(self.initial_value, f)
      f.close()

class InitialCurDatasetSaver (InitialSaver):
  def __init__ (self, base_path, original_dataset_path, chunk_size):
    super(InitialCurDatasetSaver, self).__init__(base_path)
    self.original_dataset_path = original_dataset_path
    self.chunk_size = chunk_size

  def initial_save (self):
    DatasetShuffler(self.original_dataset_path, self.chunk_size).shuffle(os.path.join(self.base_path, 'cur_dataset.h5'))

class InitialLossSaver (InitialSaver):
  def __init__ (self, base_path, maxshape, chunks):
    super(InitialLossSaver, self).__init__(base_path)
    self.maxshape = maxshape
    self.chunks = chunks
  
  def initial_save (self):
    f = h5py.File(os.path.join(self.base_path, 'loss.h5'), 'w')
    f.create_dataset('loss', (0, ), dtype='float32', chunks=self.chunks, maxshape=self.maxshape)
    f.close()

class InitialTransducerSaver (InitialSaver):
  def __init__ (self, base_path, transducer):
    super(InitialTransducerSaver, self).__init__(base_path)
    self.transducer = transducer

  def initial_save (self):
    self.transducer.save_weights(
      os.path.join(
        self.base_path,
        'checkpoints/ep{}_no{}/ckpt'.format(0, 0)
      )
    )

# initial loader

class ModelMetaLoader:
  @staticmethod
  def load (base_path):
    with open(os.path.join(base_path, 'model_meta.pickle'), 'rb') as f:
      model_meta = pickle.load(f)
      f.close()
    return model_meta
