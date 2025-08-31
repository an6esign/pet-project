from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Prediction:
    label_id: int
    label_name: str
    proba: Optional[List[float]]= None # у LinearSVC будет None