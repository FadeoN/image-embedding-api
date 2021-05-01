from typing import List
from pydantic import BaseModel, Field


class KeypointDTO(BaseModel):
    name: str
    score: float
    x: float
    y: float

    def convert_to_poem_format(self):
        return self.y, self.x, self.score


class FrameKeypointDTO(BaseModel):
    keypoints: List[KeypointDTO]


class VideoEmbeddingRequest(BaseModel):
    width: int = Field(ge=1, description="Width must be more than zero")
    height: int = Field(ge=1, description="Height must be more than zero")
    frames: List[FrameKeypointDTO]

