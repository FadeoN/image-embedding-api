from typing import List

from infrastructure.controller.model.video_embedding_request import FrameKeypointDTO


class GetVideoEmbeddingQuery:
    def __init__(self,
                 width: int,
                 height: int,
                 frames: List[FrameKeypointDTO]):
        self.width = width
        self.height = height
        self.frames = frames

    def convert_to_poem_format(self):

        joint_order = ["nose", "leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
                  "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"]

        poem_frames = []
        for frame in self.frames:

            joint_map = {keypoint.name: keypoint.convert_to_poem_format() for keypoint in frame.keypoints}
            poem_frames.append([joint_map[joint] for joint in joint_order])

        return poem_frames
