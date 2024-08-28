from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, TypeAlias, Callable, Any

import numba
import numpy as np
from mesa import Agent

if TYPE_CHECKING:
  from base.model import HKModel
  from base.tweet import TweetRecord

FollowEventRecord: TypeAlias = Tuple[int, int]  # unfollow, follow

_RECORD: TweetRecord = (-1, -1, -1)

# @numba.jit(nopython=True)
def partition_tweets(
    opinion: float,
    neighbors: Optional[List[TweetRecord]],
    recommended: Optional[List[List[TweetRecord]]],
    epsilon: float,
):
  concordant_neighbor: List[TweetRecord] = [_RECORD]
  concordant_recommended: List[TweetRecord] = [_RECORD]
  discordant_neighbor: List[TweetRecord] = [_RECORD]
  discordant_recommended: List[TweetRecord] = [_RECORD]

  if neighbors is not None:
    for a in neighbors:
      o = a[-1]
      if abs(opinion - o) <= epsilon:
        concordant_neighbor.append(a)
      else:
        discordant_neighbor.append(a)
  
  if recommended is not None:
    for a in recommended:
      o = a[-1]
      if abs(opinion - o) <= epsilon:
        concordant_recommended.append(a)
      else:
        discordant_recommended.append(a)

  sums = []
  for c in [concordant_neighbor, concordant_recommended, discordant_neighbor, discordant_recommended]:
    _sum = 0
    for _, __, t in c[1:]:
      _sum += t - opinion
    sums.append(_sum)
  sum_n = sums[0]
  sum_r = sums[1]
  sum_nd = sums[2]
  sum_rd = sums[3]

  return concordant_neighbor[1:], concordant_recommended[1:], \
      discordant_neighbor[1:], discordant_recommended[1:], \
      sum_n, sum_r, sum_nd, sum_rd


# @numba.jit(nopython=True)
def hk_agent_step(
    uid: int,
    opinion: float,
    cur_step: int,

    decay: float,
    gamma: float,
    tolerance: float,
    r_retweet: float,

    neighbors: Optional[List[TweetRecord]],
    recommended: Optional[List[TweetRecord]],

    collect_nr_agents: bool,
    collect_op_sum_agents: bool,

    report_view_tweets: bool,
    report_retweet: bool,
    report_rewiring: bool,
):

  # collect vars
  nr_agents = (0, 0, 0, 0)
  op_sum_agents = (0, 0, 0, 0)
  e_view_tweets: Optional[List[List[TweetRecord]]] = None
  e_retweet: Optional[TweetRecord] = None
  e_rewiring: Optional[FollowEventRecord] = None

  # return vars
  next_opinion = opinion
  next_tweet: Optional[TweetRecord] = None
  next_follow: Optional[FollowEventRecord] = None

  # calculate tweet sets

  concordant_neighbor, concordant_recommended, \
      discordant_neighbor, discordant_recommended, \
      sum_n, sum_r, sum_nd, sum_rd = partition_tweets(
          opinion, neighbors, recommended, tolerance
      )

  n_neighbor = len(concordant_neighbor)
  n_recommended = len(concordant_recommended)
  n_concordant = n_neighbor + n_recommended

  # collect
  if collect_nr_agents:
    nr_agents = (n_neighbor, n_recommended, len(
        discordant_neighbor), len(discordant_recommended))
  if report_view_tweets:
    e_view_tweets = [
            concordant_neighbor, concordant_recommended,
            discordant_neighbor, discordant_recommended,
        ]

  # @@ influence
  # the agent reads all concordant tweets
  if n_concordant > 0:
    next_opinion += ((sum_r + sum_n) /
                     n_concordant) * decay
  # else, the opinion does not change

  # collect
  if collect_op_sum_agents:
    op_sum_agents = (sum_n, sum_r, sum_nd, sum_rd)

  # generate random numbers
  rnd_retweet, rnd_rewiring = np.random.uniform(low=0, high=1, size=(2,))

  # @@ tweet or retweet
  if n_neighbor > 0 and rnd_retweet < r_retweet:  # randomly retweet one
    retweet_index = int(n_concordant * rnd_retweet /
                        r_retweet) % n_concordant
    if retweet_index < n_neighbor:
      next_tweet = concordant_neighbor[retweet_index]
    else:
      next_tweet = concordant_recommended[retweet_index - n_neighbor]
    # collect
    if report_retweet:
      e_retweet = next_tweet
  else:  # post a new one
    next_tweet = (uid, cur_step, next_opinion)
    
  # @@ rewiring
  if gamma > 0 \
      and discordant_neighbor and concordant_recommended \
          and rnd_rewiring < gamma:
    idx1 = np.random.randint(low=0, high=len(concordant_recommended))
    idx2 = np.random.randint(low=0, high=len(discordant_neighbor))
    follow, __, _ = concordant_recommended[idx1]
    unfollow, __, _ = discordant_neighbor[idx2]
    next_follow = (unfollow, follow)

  
  # collect
  if report_rewiring and next_follow is not None:
    unfollow, follow = next_follow
    e_rewiring = next_follow
    
  # damn numba
  r1: Tuple[float, float, TweetRecord] = (next_opinion, next_tweet, next_follow)
  r2: Tuple[List[int], List[float]] = (nr_agents, op_sum_agents)
  r3 = (e_view_tweets, e_retweet, e_rewiring)
  
  return r1, r2, r3

# r = hk_agent_step(0, 0, 0, 1, 1, 1, 1, [(1, 1, 1), (-1, -1, -1)], None, True, True, True, True, True, )

_EVENT_NAMES = [
  'view_tweets',
  'retweet',
  'rewiring',
]

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
    self.cur_tweet: Optional[TweetRecord] = None

    # future state
    self.next_opinion = self.cur_opinion
    self.next_tweet = self.cur_tweet
    self.next_follow: Optional[FollowEventRecord] = None

    # recorded data
    self.nr_agents = [0, 0, 0, 0]
    self.op_sum_agents = [0, 0, 0, 0]

    # event logging
    has_event_logger = self.model.event_logger is not None
    self.report_view_tweets = has_event_logger and 'e:view_tweets' in self.model.collect
    self.report_rewiring = has_event_logger and 'e:rewiring' in self.model.collect
    self.report_retweet = has_event_logger and 'e:retweet' in self.model.collect

    self.step_dict = dict(
        uid=self.unique_id,
        # opinion: float,
        # cur_step: int,

        decay=self.model.p.decay,
        gamma=self.model.p.rewiring_rate,
        tolerance=self.model.p.tolerance,
        r_retweet=self.model.p.retweet_rate,

        # neighbors: List[TweetRecord],
        # recommended: List[TweetRecord],

        collect_nr_agents='nr_agents' in self.model.collect,
        collect_op_sum_agents='op_sum_agents' in self.model.collect,

        report_view_tweets=self.report_view_tweets,
        report_retweet=self.report_retweet,
        report_rewiring=self.report_rewiring,
    )

  def step(self):

    # get the neighbors
    # TODO change model codes
    neighbors: List[TweetRecord] = self.model.grid.get_neighbors(
        self.unique_id, include_center=False)
    recommended: List[TweetRecord] = self.model.get_recommendation(
        self, neighbors)

    # call with acceleration
    return_params, collect_params, events = hk_agent_step(
        **self.step_dict,
        opinion=self.cur_opinion,
        cur_step=self.model.cur_step,
        neighbors=neighbors,
        recommended=recommended,
    )

    self.next_opinion, self.next_tweet, self.next_follow = return_params
    self.nr_agents, self.op_sum_agents = collect_params

    for e, n in zip(events, _EVENT_NAMES):
      if e is not None:
        self.model.event_logger(dict(
          name = n,
          uid = self.unique_id,
          step = self.model.cur_step,
          body = e,
        ))
