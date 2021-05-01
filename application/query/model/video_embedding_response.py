from typing import List

from domain.model.image import FrameVectorPair


class VideoEmbeddingResponse:
    def __init__(
        self,
        frameVectorPairs: List[FrameVectorPair]
    ):
        self.frameVectorPairs = frameVectorPairs
