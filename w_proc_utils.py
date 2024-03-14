from typing import List
from numpy.typing import NDArray

import numpy as np


def first_less_than(arr: NDArray, k: float):
  mask = arr < k
  idx = np.argmax(mask)
  return idx if mask[idx] else arr.size


def moving_average(data: NDArray, window_size: int):
  if window_size < 2:
    return data
  pad_width = window_size // 2
  pad_data = np.pad(data, pad_width, mode='edge')
  window = np.ones(window_size) / window_size
  moving_avg = np.convolve(pad_data, window, 'valid')
  return moving_avg


def proc_opinion_diff(dn: NDArray, dr: NDArray, average=3, error=1e-4):

  dn[0] = dn[1]
  dr[0] = dr[1]
  sn = moving_average(np.std(dn, axis=1), average)
  sr = moving_average(np.std(dr, axis=1), average)
  an = moving_average(np.mean(dn, axis=1), average)
  ar = moving_average(np.mean(dr, axis=1), average)
  
  pad_index = min(
    first_less_than(sn, error),
    first_less_than(sr, error),
  )
  
  ratio_s = sr[:pad_index] / (sr[:pad_index] + sn[:pad_index])
  
  return sn, sr, an, ar, ratio_s
  
  
