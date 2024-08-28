from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, TypeAlias

import numpy as np
from mesa import Agent

if TYPE_CHECKING:
  from base.model import HKModel

class HKAgent(Agent):

  model: 'HKModel' = None

  def __init__(
      self,
      unique_id: int,
      model: 'HKModel',
      opinion: Optional[float] = None
  ):
    super().__init__(unique_id, model)

    # current state
    self.cur_opinion = opinion if opinion is not None else np.random.uniform(
        -1, 1)

    # future state
    self.next_opinion = self.cur_opinion
    self.next_follow: Optional[Tuple['HKAgent', 'HKAgent']] = None

    # recorded data
    self.nr_agents = [0, 0, 0, 0]
    self.op_sum_agents = [0, 0, 0, 0]
    self.follow_event = [False, -1, -1]

  def step(self):
    # clear data
    self.next_opinion = self.cur_opinion
    self.next_follow = None

    # get the neighbors
    neighbors: List['HKAgent'] = self.model.grid.get_neighbors(
        self.unique_id, include_center=False)
    recommended: List['HKAgent'] = self.model.get_recommendation(
        self, neighbors)
    if not (len(neighbors) + len(recommended)):
      return

    # calculate concordant set
    epsilon = self.model.p.tolerance
    gamma = self.model.p.rewiring_rate

    concordant_neighbor: List['HKAgent'] = []
    concordant_recommended: List['HKAgent'] = []
    discordant_neighbor: List['HKAgent'] = []
    discordant_recommended: List['HKAgent'] = []

    for a in neighbors:
      if abs(self.cur_opinion - a.cur_opinion) <= epsilon:
        concordant_neighbor.append(a)
      else:
        discordant_neighbor.append(a)
    for a in recommended:
      if abs(self.cur_opinion - a.cur_opinion) <= epsilon:
        concordant_recommended.append(a)
      else:
        discordant_recommended.append(a)

    # update value
    n_neighbor = len(concordant_neighbor)
    n_recommended = len(concordant_recommended)
    n_concordant = n_neighbor + n_recommended

    if 'nr_agents' in self.model.collect:
      self.nr_agents = [n_neighbor, n_recommended, len(
          discordant_neighbor), len(discordant_recommended)]

    sum_n = 0
    sum_r = 0
    if n_concordant > 0:
      sum_n = sum(a.cur_opinion -
                  self.cur_opinion for a in concordant_neighbor)
      sum_r = sum(a.cur_opinion -
                  self.cur_opinion for a in concordant_recommended)

      self.next_opinion += ((sum_r + sum_n) /
                            n_concordant) * self.model.p.decay

    if 'op_sum_agents' in self.model.collect:
      self.op_sum_agents = [
          sum_n,
          sum_r,
          sum(a.cur_opinion - self.cur_opinion for a in discordant_neighbor),
          sum(a.cur_opinion - self.cur_opinion for a in discordant_recommended)
      ]

    # handle rewiring
    if gamma > 0 and discordant_neighbor and concordant_recommended and np.random.uniform() < gamma:
      follow = np.random.choice(concordant_recommended)
      unfollow = np.random.choice(discordant_neighbor)
      self.next_follow = (unfollow, follow)

    if 'follow_event' in self.model.collect:
      self.follow_event = [True, self.next_follow[0].unique_id, self.next_follow[1].unique_id] \
          if self.next_follow is not None else [False, -1, -1]
