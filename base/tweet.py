from typing import Tuple, TypeAlias

import dataclasses

TweetRecord: TypeAlias = Tuple[int, int, float]

@dataclasses.dataclass
class Tweet:
  uid: int
  step: int
  opinion: float
  
  def to_record(self) -> TweetRecord:
    return (self.uid, self.step, self.opinion)
  