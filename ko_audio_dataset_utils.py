import re
import numpy as np
import librosa

def sentence_filter(raw_sentence):
  return special_filter(bracket_filter(raw_sentence))

def special_filter(sentence):
  SENTENCE_MARK = ['?', '!']
  NOISE = ['o', 'n', 'u', 'b', 'l']
  EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', '.', ',']
  
  new_sentence = str()
  for idx, ch in enumerate(sentence):
    if ch not in SENTENCE_MARK:
      # o/, n/ 등 처리
      if idx + 1 < len(sentence) and ch in NOISE and sentence[idx+1] == '/': 
        continue 

    if ch == '#': 
      new_sentence += '샾'

    elif ch not in EXCEPT: 
      new_sentence += ch

  pattern = re.compile(r'\s\s+')
  new_sentence = re.sub(pattern, ' ', new_sentence.strip())
  return new_sentence


def bracket_filter(sentence):
  new_sentence = str()
  flag = False
  
  for ch in sentence:
    if ch == '(' and flag == False:
      flag = True
      continue
    if ch == '(' and flag == True:
      flag = False
      continue
    if ch != ')' and flag == False:
      new_sentence += ch
  return new_sentence

chosung = ("ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")

jungsung = ("ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ")

jongsung = ("", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")

def isHangeul(one_character):
  return 0xAC00 <= ord(one_character[:1]) <= 0xD7A3

def hangeulExplode(one_hangeul):
  a = one_hangeul[:1]
  if isHangeul(a) != True:
    return False
  b = ord(a) - 0xAC00
  cho = b // (21*28)
  jung = b % (21*28) // 28
  jong = b % 28
  if jong == 0:
    return (chosung[cho], jungsung[jung])
  else:
    return (chosung[cho], jungsung[jung], jongsung[jong])

def hangeulJoin(inputlist):
  result = ""
  cho, jung, jong = 0, 0, 0
  inputlist.insert(0, "")
  while len(inputlist) > 1:
    if inputlist[-1] in jongsung:
      if inputlist[-2] in jungsung:
        jong = jongsung.index(inputlist.pop())
      
      else:
        result += inputlist.pop()
    elif inputlist[-1] in jungsung:
      if inputlist[-2] in chosung:
        jung = jungsung.index(inputlist.pop())
        cho = chosung.index(inputlist.pop())
        result += chr(0xAC00 + ((cho*21)+jung)*28+jong)
        cho, jung, jong = 0, 0, 0
      else:
        result += inputlist.pop()

    else:
      result += inputlist.pop()
  else:
    return result[::-1]

def pureosseugi(inputtext):
  result = ""
  for i in inputtext:
    if isHangeul(i) == True:
      for j in hangeulExplode(i):
        result += j
    else:
      result += i
  
  return result

def moasseugi(inputtext):
  t1 = []
  for i in inputtext:
    t1.append(i)

  return hangeulJoin(t1)

def extract_spectrogram (path, sr=16000, n_fft=320, hop_len=160, n_mels=80):
  # waveform = np.memmap(path, dtype='h', mode='r')

  with open(path, 'rb') as f:
    bytes = f.read()
    f.close()
  waveform = np.frombuffer(bytes[0: len(bytes) - len(bytes) % 2], dtype=np.int16)

  S = librosa.feature.melspectrogram(y=np.float32(waveform), n_fft=n_fft, hop_length=hop_len, window='hamming', sr=sr, n_mels=n_mels)
  log_S = librosa.power_to_db(S, ref=np.max)
  log_S = np.transpose(log_S)

  return log_S

def check_skip_label (label):
  for hangul_phoneme in label:
    if check_skip_hangul_phoneme(hangul_phoneme):
      return True
  
  return False

def check_skip_hangul_phoneme (hangul_phoneme):
  return not isHangeul(hangul_phoneme) and not hangul_phoneme == ' ' and not hangul_phoneme == '?'

def vector_to_sentence (phoneme_dict, x):
  return ''.join([phoneme_dict.phonemes[idx] for idx in x])
