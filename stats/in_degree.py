import sys
import os

from collections import Counter
from typing import Dict, Union, Optional
import numpy as np
import networkx as nx
from numpy.typing import NDArray

import powerlaw
from contextlib import contextmanager


@contextmanager
def suppress_output():
  with open(os.devnull, 'w') as devnull:
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull
    try:
      yield
    finally:
      sys.stdout = old_stdout
      sys.stderr = old_stderr


class InDegreeCollector:

  def __init__(
      self,
      dist2: Optional[str] = None,
      full_data: bool = False,
  ) -> None:
    self.dist2 = dist2 or 'exponential'
    self.full_data = full_data

  def collect(
          self,
          prefix: str,
          digraph: nx.DiGraph, 
          *args, **kwargs) -> Dict[str, Union[float, NDArray]]:

    in_degree_dict = dict(digraph.in_degree())

    with suppress_output():
      p_res = powerlaw.Fit(
          np.array([v for v in in_degree_dict.values() if v > 0]).flatten())
      R, p = p_res.distribution_compare(
          'power_law', self.dist2, normalized_ratio=True)

    ret_dict = {
        prefix + '-alpha': p_res.alpha,
        prefix + '-sigma': p_res.sigma,
        prefix + '-xmin': p_res.xmin,
        prefix + '-xmax': p_res.xmax,
        prefix + '-R': R,
        prefix + '-p-value': p
    }

    if self.full_data:
      in_degree_dist = Counter(in_degree_dict.values())
      in_degree_val = np.array(sorted(in_degree_dist.items())).T
      ret_dict[prefix] = in_degree_val

    return ret_dict
