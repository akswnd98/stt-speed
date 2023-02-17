from trainer import SimpleInitialSaver, SimpleGenerator, SimpleLoader
import tensorflow as tf
from phoneme_dict import PhonemeDictLoader

if __name__ == '__main__':
  LEARNING_RATE = 0.0001
  EPOCHS = 20
  BATCH_SIZE = 8
  BATCH_NUM = 4
  BASE_PATH = 'saves/train_state7'
  ORIGINAL_DATASET_PATH = 'saves/dataset_large.h5'
  CHUNK_SIZE = 20000
  SAVE_TIME_DELTA = 10000
  N_MELS = 80

  EMBEDDING_DIM = 128
  UNITS = 1024
  CODER_OUTPUT_DIM = 320
  JOINT_NET_INNER_DIM = 320
  MODEL_META = (
    EMBEDDING_DIM,
    UNITS,
    CODER_OUTPUT_DIM,
    JOINT_NET_INNER_DIM,
    len(PhonemeDictLoader('saves/phoneme_dict.pickle').phonemes)
  )

  # SimpleInitialSaver.initial_save(BASE_PATH, ORIGINAL_DATASET_PATH, CHUNK_SIZE, MODEL_META, 400000, 200000)

  # trainer = SimpleGenerator.generate(
  #   LEARNING_RATE,
  #   EPOCHS,
  #   BATCH_SIZE * BATCH_NUM,
  #   BATCH_SIZE,
  #   BATCH_NUM,
  #   BASE_PATH,
  #   ORIGINAL_DATASET_PATH,
  #   CHUNK_SIZE,
  #   SAVE_TIME_DELTA,
  #   MODEL_META,
  #   N_MELS
  # )

  trainer = SimpleLoader.load(
    BASE_PATH,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE * BATCH_NUM,
    BATCH_SIZE,
    BATCH_NUM,
    ORIGINAL_DATASET_PATH,
    CHUNK_SIZE,
    SAVE_TIME_DELTA,
    N_MELS
  )

  with tf.device('/gpu:0'):
    trainer.train()
