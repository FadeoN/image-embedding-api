from typing import List


class FrameVectorPair:
    def __init__(self, order: int,
                 vector: List[float]):
        self.order = order
        self.vector = vector