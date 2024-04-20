from typing import Optional, Union
from numpy.typing import NDArray

import logging

import numpy as np


def get_logger(name: str, path: str):

  logger = logging.getLogger(name)

  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.DEBUG)

  file_handler = logging.FileHandler(path)
  file_handler.setLevel(logging.INFO)

  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

  console_handler.setFormatter(formatter)
  file_handler.setFormatter(formatter)

  logger.addHandler(console_handler)
  logger.addHandler(file_handler)

  logger.setLevel(logging.DEBUG)
  
  return logger


def first_less_than(arr: NDArray, k: float):
  mask = arr < k
  idx = np.argmax(mask)
  return idx if mask[idx] else arr.size

def first_more_or_equal_than(arr: NDArray, k: float):
  mask = arr > k
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


def proc_opinion_diff(dn: NDArray, dr: NDArray, 
  n_n: NDArray, n_r: NDArray, average=3, error=1e-4):

  if dn.shape[0] > 1:
    dn[0] = dn[1]
    dr[0] = dr[1]
    n_n[0] = n_n[1]
    n_r[0] = n_r[1]
  dan = np.abs(dn)
  dar = np.abs(dr)
  sn = moving_average(np.std(dan, axis=1), average)
  sr = moving_average(np.std(dar, axis=1), average)
  an = moving_average(np.mean(dan, axis=1), average)
  ar = moving_average(np.mean(dar, axis=1), average)
  
  sum_n1 = moving_average(np.mean(n_n, axis=1), average)
  sum_r1 = moving_average(np.mean(n_r, axis=1), average)
  
  pad_index = min(
    first_less_than(sn, error),
    first_less_than(sr, error),
  )
  
  ratio_s = sr[:pad_index] / (sr[:pad_index] + sn[:pad_index])
  ratio_a = ar[:pad_index] / (ar[:pad_index] + an[:pad_index])
  
  return sn, sr, an, ar, sum_n1, sum_r1, ratio_s, ratio_a
  


def proc_opinion_ratio(
  
  sum_n: NDArray, sum_r: NDArray, 
  n_n: NDArray, n_r: NDArray, 
  error=1e-2):

  sum_an = np.abs(sum_n)
  sum_ar = np.abs(sum_r)
  
  sum_mask = np.logical_and(
    sum_an > error,
    sum_ar > error
  )
  n_mask = (n_n + n_r) > 1
  
  ratio_n = n_r[n_mask] / (n_n + n_r)[n_mask]
  
  n_n1 = np.array(n_n)
  n_n1[n_n1 == 0] = 1
  n_nr1 = np.array(n_n + n_r)
  n_nr1[n_nr1 == 0] = 1
  
  sum_nom = (sum_ar + sum_an) / n_nr1
  sum_den = (sum_an) / n_n1
  
  ratio_sum = sum_nom[sum_mask] / sum_den[sum_mask]
  ratio_sum_log = np.log(ratio_sum) / np.log(10)

  return ratio_n, ratio_sum_log
  


def first_index_above_min(arr: np.ndarray, error: float = 1e-5) -> int:
  min_val = np.min(arr)
  threshold = min_val + error
  for i in range(arr.size - 1, -1, -1):
    if arr[i] > threshold:
      return i
  return 0


def area_under_curve(points: NDArray, arg_sort = False, complement: Optional[float]=1):
  # points: (n, 2)
  
  x = points[0]
  y = points[1]
  if arg_sort:
    indices = np.argsort(x)
    x = x[indices]
    y = y[indices]
  
  dx = np.diff(x)
  area = 0.5 * np.sum(dx * (y[1:] + y[:-1]))
  if complement:
    area += (complement - x[-1]) * y[-1]
  
  return area