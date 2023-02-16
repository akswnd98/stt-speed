from decoder import AudioDecoder
import ko_audio_dataset_utils as kadu
from phoneme_dict import PhonemeDictLoader
from model import Transducer

EMBEDDING_DIM = 128
UNITS = 1024
CODER_OUTPUT_DIM = 512
JOINT_NET_INNER_DIM = 512

mel_spec = kadu.extract_spectrogram(
  '../stt/ko-audio-dataset/평가용_데이터/KsponSpeech_eval/eval_clean/KsponSpeech_E00034.pcm',
  sr=16000,
  n_fft=512,
  hop_len=256,
  n_mels=128
)

phoneme_dict = PhonemeDictLoader('saves/phoneme_dict.pickle')
transducer = Transducer(EMBEDDING_DIM, UNITS, CODER_OUTPUT_DIM, JOINT_NET_INNER_DIM, len(phoneme_dict.phonemes))
transducer.load_weights('saves/train_state3/checkpoints/ep13_no50080/ckpt')
decoder = AudioDecoder(transducer)

print(kadu.moasseugi(kadu.vector_to_sentence(phoneme_dict, decoder.decode(mel_spec)[1: ])))
