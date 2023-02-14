import time
import multiprocessing as mp
import numpy as np
from functools import reduce

def acc (ll):
  ret = 0
  for l in ll:
    for v in l:
      ret += v
  return ret

if __name__ == '__main__':
  start = time.time()
  ll = [[1 for _ in range(10000)] for _ in range(10000)]
  pool = mp.Pool(10)
  result = pool.map(acc, [ll for _ in range(10)])
  end = time.time()
  print('{} {}'.format(end - start, result))

  # start = time.time()
  # ll = [[1 for _ in range(10000)] for _ in range(10000)]
  # result = list(map(acc, [ll for _ in range(100)]))
  # end = time.time()
  # print('{} {}'.format(end - start, result))
