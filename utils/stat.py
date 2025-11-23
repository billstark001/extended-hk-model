from numpy.typing import NDArray
from typing import Optional, Tuple, Callable, TypeVar, List, Literal, cast, Any
from scipy.interpolate import interp1d
from collections import OrderedDict, deque
import numpy as np


# ============ 数组查找工具 ============
def first_less_than(arr: NDArray, k: float) -> int:
    mask = arr < k
    idx = np.argmax(mask)
    return int(idx) if mask[idx] else int(arr.size)


def last_less_than(arr: NDArray, k: float) -> int:
    return int(np.argmin(arr < k)) - 1


def first_more_or_equal_than(arr: NDArray, k: float) -> int:
    mask = arr >= k
    idx = np.argmax(mask)
    return int(idx) if mask[idx] else int(arr.size)


def first_index_above_min(arr: NDArray, error: float = 1e-5) -> int:
    threshold = np.min(arr) + error
    for i in range(arr.size - 1, -1, -1):
        if arr[i] > threshold:
            return i
    return 0


# ============ 统计工具 ============
def moving_average(
    data: NDArray,
    window_size: int,
    pad_mode: str = "edge",
    convolve_mode: Literal["valid", "same", "full"] = "valid",
) -> NDArray:
    if window_size < 2:
        return data
    pad_width = window_size // 2
    pad_data = np.pad(data, pad_width, mode=cast(Any, pad_mode))
    window = np.ones(window_size) / window_size
    return np.convolve(pad_data, window, convolve_mode)


def gaussian_kernel(x: NDArray, h: float) -> NDArray:
    return np.exp(-0.5 * (x / h) ** 2) / (np.sqrt(2 * np.pi) * h)


def compute_kde_density(
    x_grid: NDArray, x_data: NDArray, h: float, epsilon: float = 1e-14
) -> NDArray:
    """计算核密度估计"""
    diff = x_grid.reshape(-1, 1) - x_data.reshape(1, -1)
    kde = np.sum(gaussian_kernel(diff, h), axis=1)
    return kde / (x_data.size * h) + epsilon


def compute_weighted_stats(
    x_result: NDArray,
    x_data: NDArray,
    y_data: NDArray,
    bandwidths: NDArray,
    epsilon: float = 1e-14,
) -> Tuple[NDArray, NDArray]:
    """计算加权均值和方差"""
    diff = x_result.reshape(-1, 1) - x_data.reshape(1, -1)
    weights_unnorm = gaussian_kernel(diff, cast(float, bandwidths.reshape(-1, 1)))
    weights = weights_unnorm / (np.sum(weights_unnorm, axis=1, keepdims=True) + epsilon)

    means = np.sum(weights * y_data.reshape(1, -1), axis=1)
    means2 = np.sum(weights * (y_data.reshape(1, -1) ** 2), axis=1)
    variances = np.maximum(means2 - means**2, 0)

    return means, variances


def adaptive_moving_stats(
    x: NDArray,
    y: NDArray,
    h0: float,
    alpha: float = 0.5,
    g: float = 1.0,
    min: Optional[float] = None,
    max: Optional[float] = None,
    density_x: Optional[NDArray] = None,
    result_x: Optional[NDArray] = None,
    density_estimation_point: Optional[int] = 20,
    result_point: Optional[int] = 100,
    epsilon: float = 1e-14,
    edge_padding: float = 0.0,
    edge_method: str = "reflect",
    min_bandwidth: Optional[float] = None,
    max_bandwidth: Optional[float] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    自适应移动窗口统计 (KDE + 自适应带宽)

    h(x) = h0 * (ρ(x) / g)^(-α)
    """
    min_val = min if min is not None else x.min()
    max_val = max if max is not None else x.max()

    # 边缘处理
    if edge_padding > 0:
        range_size = max_val - min_val
        padding = range_size * edge_padding
        min_padded, max_padded = min_val - padding, max_val + padding

        if edge_method == "reflect":
            left_mask = x < (min_val + padding)
            right_mask = x > (max_val - padding)
            x_extended = np.concatenate(
                [2 * min_val - x[left_mask], x, 2 * max_val - x[right_mask]]
            )
            y_extended = np.concatenate([y[left_mask], y, y[right_mask]])
        else:
            x_extended, y_extended = x, y
    else:
        min_padded, max_padded = min_val, max_val
        x_extended, y_extended = x, y

    # 密度估计点
    if density_x is None:
        density_x_pts = (
            x_extended
            if density_estimation_point is None
            else np.linspace(
                min_padded, max_padded, density_estimation_point, dtype=float
            )
        )
    else:
        density_x_pts = density_x

    density = compute_kde_density(density_x_pts, x_extended, h0, epsilon)

    # 结果点
    if result_x is None:
        result_x_pts = (
            x
            if result_point is None
            else np.linspace(min_val, max_val, result_point, dtype=float)
        )
    else:
        result_x_pts = result_x

    # 插值密度并计算自适应带宽
    fill_value: Any = (
        "extrapolate"
        if edge_method in ["extrapolate", "none"]
        else (density[0], density[-1])
    )
    density_interp = interp1d(
        density_x_pts, density, fill_value=fill_value, bounds_error=False
    )
    density_resampled = np.maximum(density_interp(result_x_pts), epsilon)

    h_t = h0 * (density_resampled / g) ** (-alpha)
    if min_bandwidth is not None:
        h_t = np.maximum(h_t, min_bandwidth)
    if max_bandwidth is not None:
        h_t = np.minimum(h_t, max_bandwidth)

    means, variances = compute_weighted_stats(
        result_x_pts, x_extended, y_extended, h_t, epsilon
    )

    return result_x_pts, means, variances


# ============ 几何计算 ============
def area_under_curve(
    points: NDArray, arg_sort: bool = False, complement: Optional[float] = 1
) -> float:
    x, y = points[0], points[1]
    if arg_sort:
        indices = np.argsort(x)
        x, y = x[indices], y[indices]

    area = 0.5 * np.sum(np.diff(x) * (y[1:] + y[:-1]))
    if complement:
        area += (complement - x[-1]) * y[-1]
    return area


# ============ 自适应采样 ============
T = TypeVar("T")


def adaptive_discrete_sampling(
    f: Callable[[int], T],
    error_threshold: float,
    t_start: int,
    t_end: int,
    max_interval: int | None = None,
    err_func: Callable[[T, T, T, float], float] | None = None,
) -> Tuple[List[int], List[T]]:
    """离散时间轴自适应采样"""
    if t_start >= t_end:
        raise ValueError("t_start must be less than t_end")

    samples = OrderedDict({t_start: f(t_start), t_end: f(t_end)})
    queue: deque[Tuple[int, int]] = deque([(t_start, t_end)])

    while queue:
        t_left, t_right = queue.popleft()
        if t_right - t_left <= 1:
            continue

        force_sample = max_interval is not None and (t_right - t_left) > max_interval
        t_mid = (
            t_left + (max_interval or 0) if force_sample else (t_left + t_right) // 2
        )

        if t_mid in [t_left, t_right] or t_mid in samples:
            continue

        f_left, f_right, f_mid = samples[t_left], samples[t_right], f(t_mid)
        t_mid_rate = (t_mid - t_left) / (t_right - t_left)

        if err_func:
            error = err_func(f_left, f_right, f_mid, t_mid_rate)
        else:
            f_interp = cast(Any, f_right) * t_mid_rate + cast(Any, f_left) * (
                1 - t_mid_rate
            )
            error = abs(cast(Any, f_mid) - f_interp)

        if error > error_threshold or force_sample:
            samples[t_mid] = f_mid
            queue.extend([(t_left, t_mid), (t_mid, t_right)])

    t_arr = sorted(samples.keys())
    return t_arr, [samples[t] for t in t_arr]


# ============ 数据合并 ============
def merge_data_with_axes(
    *data: Tuple[NDArray, NDArray]
) -> Tuple[NDArray, List[NDArray]]:
    x_merged = np.unique(np.concatenate([x for x, _ in data]))
    y_list = [
        interp1d(
            x,
            y,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=cast(Any, "extrapolate"),
        )(x_merged)
        for x, y in data
    ]
    return x_merged, y_list


# ============ 力场和势能估计 ============
def estimate_force_field_kde(
    x_data: NDArray, dx_data: NDArray, x_grid: NDArray, h: float, k: float = 1.0
) -> NDArray:
    """使用KDE估计力场 F(x) = Σ K_h(x - x_i) · F_i / Σ K_h(x - x_i)"""
    diff = x_grid.reshape(-1, 1) - x_data.reshape(1, -1)
    weights = gaussian_kernel(diff, h)
    F_i = k * dx_data
    return np.sum(weights * F_i.reshape(1, -1), axis=1) / (
        np.sum(weights, axis=1) + 1e-14
    )


def estimate_potential_from_force(x_grid: NDArray, F_grid: NDArray) -> NDArray:
    """梯形积分估计势能 V(x) = -∫ F(x) dx"""
    dx = np.diff(x_grid)
    integral_segments = dx * (F_grid[1:] + F_grid[:-1]) / 2
    V_grid = np.zeros_like(F_grid)
    V_grid[1:] = -np.cumsum(integral_segments)
    return V_grid
