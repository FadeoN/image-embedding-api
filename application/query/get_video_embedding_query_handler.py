from typing import List

from application.query.get_video_embedding_query import GetVideoEmbeddingQuery
from application.query.model.video_embedding_response import VideoEmbeddingResponse
from domain.model.image import FrameVectorPair
from poem.pr_vipe import infer


async def handle(query: GetVideoEmbeddingQuery) -> VideoEmbeddingResponse:

    frames = query.convert_to_poem_format()

    vectors = infer.infer(query.width, query.height, frames)
    frame_vector_pairs = [FrameVectorPair(order=idx, vector=vector) for idx, vector in enumerate(vectors)]

    return VideoEmbeddingResponse(frameVectorPairs=frame_vector_pairs)
