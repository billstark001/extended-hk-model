from typing import Tuple, TypeAlias

import dataclasses

TweetRecord: TypeAlias = Tuple[int, int, float]

@dataclasses.dataclass
class Tweet:
  user: int
  step: int
  opinion: float
  
  def to_record(self) -> TweetRecord:
    return (self.user, self.step, self.opinion)
  