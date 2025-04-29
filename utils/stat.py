from typing import Optional, Tuple, Callable, TypeVar
from numpy.typing import NDArray, DTypeLike
from scipy.interpolate import interp1d

import logging

import numpy as np
import zlib
import base64
import numpy as np


def compress_array_to_b64(arr: NDArray) -> str:
  bytes_data = arr.tobytes()
  compressed_data = zlib.compress(bytes_data)
  b64_compressed_data = base64.b64encode(compressed_data)
  return b64_compressed_data.decode('utf-8')


def decompress_b64_to_array(b64_str: str, dtype: DTypeLike) -> NDArray:
  compressed_data = base64.b64decode(b64_str.encode('utf-8'))
  bytes_data = zlib.decompress(compressed_data)
  arr = np.frombuffer(bytes_data, dtype=dtype)

  return arr


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


def last_less_than(arr: NDArray, k: float):
  mask = arr < k
  idx = np.argmin(mask)
  return idx - 1


def first_more_or_equal_than(arr: NDArray, k: float):
  mask = arr >= k
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


def adaptive_moving_stats(
  x: NDArray,
  y: NDArray,
  h0: float,
  alpha: float = 0.5,
  g: float = 1.0,
  min: Optional[float] = None,
  max: Optional[float] = None,
  density_estimation_point: Optional[int] = 20,
  result_point: Optional[int] = 100,
  epsilon=1e-14,
) -> Tuple[NDArray, NDArray, NDArray]:

  min = min if min is not None else x.min()
  max = max if max is not None else x.max()

  # estimate current point density
  density_x = x if density_estimation_point is None else np.linspace(
    min, max, density_estimation_point, dtype=float)
  kde = np.sum(
    np.exp(-0.5 * ((density_x.reshape((-1, 1)) -
         x.reshape((1, -1))) / h0) ** 2) / np.sqrt(2 * np.pi * h0 ** 2),
    axis=1
  )
  density = kde / (x.size * h0)

  # calculate result point
  result_x = x if result_point is None else np.linspace(
    min, max, result_point, dtype=float)

  # calculate adaptive width
  density_resampled_ = interp1d(
    density_x, density, fill_value='extrapolate')
  density_resampled: NDArray = density_resampled_(result_x)
  lambda_t = (density_resampled / g) ** (-alpha)
  h_t = h0 * lambda_t

  # calculate avg & std
  mask = np.exp(-0.5 * ((
    result_x.reshape((-1, 1)) - x.reshape((1, -1))
  ) / h_t.reshape((-1, 1))) ** 2) / np.sqrt(2 * np.pi * h_t.reshape((-1, 1)) ** 2)
  # shape: (result_point, #value)
  mask_sum = np.sum(mask, axis=1).reshape((-1, 1)) + epsilon
  weights = mask / mask_sum
  means = np.sum(weights * y.reshape((1, -1)), axis=1)
  means2 = np.sum(weights * (y.reshape((1, -1)) ** 2), axis=1)
  variances = means2 - means ** 2

  return result_x, means, variances


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


def area_under_curve(points: NDArray, arg_sort=False, complement: Optional[float] = 1):
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

T = TypeVar('T')

def adaptive_discrete_sampling(
  f: Callable[[int], T],
  error_threshold: float,
  t_start: int,
  t_end: int,
  max_interval: int | None = None,
  err_func: Callable[[T, T], float] | None = None
):
  """
  对离散时间轴上的函数f，进行自适应采样（非递归实现）。

  参数:
    f:    可调用对象，f(t: int) -> float
    error_threshold: 浮点数，误差上限
    t_start:   整数，采样起始时刻
    t_end:     整数，采样结束时刻
    max_interval:  整数或None，相邻采样点最大间隔（包含端点），若None则不限制
  返回:
    t_samples: numpy数组，采样的时刻（升序）
    f_samples: numpy数组，采样的函数值
  """
  if t_start >= t_end:
    raise ValueError("t_start 必须小于 t_end")

  from collections import OrderedDict, deque
  samples = OrderedDict()
  samples[t_start] = f(t_start)
  samples[t_end] = f(t_end)

  # 队列元素为 (t_left, t_right)
  interval_queue = deque()
  interval_queue.append((t_start, t_end))

  while interval_queue:
    t_left, t_right = interval_queue.popleft()
    if t_right - t_left <= 1:
      continue  # 无需再细分

    # 判断是否需要强制采样中间点（因max_interval限制）
    force_sample = False
    if max_interval is not None and (t_right - t_left) > max_interval:
      force_sample = True
      # 取离t_left最近的合法采样点
      t_mid = t_left + max_interval
    else:
      t_mid = (t_left + t_right) // 2

    if t_mid == t_left or t_mid == t_right:
      continue  # 已到极限分辨率

    if t_mid not in samples:
      f_left, f_right = samples[t_left], samples[t_right]
      if err_func is not None:
        error = abs(err_func(f_left, f_right))
      else:
        error = abs(f_left - f_right)
      # 满足误差或强制采样时，采样并细分
      if error > error_threshold or force_sample:
        samples[t_mid] = f(t_mid)
        interval_queue.append((t_left, t_mid))
        interval_queue.append((t_mid, t_right))
      # 否则，不再细分

  t_arr = sorted(samples.keys())
  f_arr = [samples[t] for t in t_arr]
  return t_arr, f_arr
