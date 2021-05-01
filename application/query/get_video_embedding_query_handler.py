from typing import List

from application.query.get_video_embedding_query import GetVideoEmbeddingQuery
from domain.model.image import ImageEmbedding
from poem.pr_vipe import infer


async def handle(query: GetVideoEmbeddingQuery) -> List[ImageEmbedding]:

    frames = query.convert_to_poem_format()

    vectors = infer.infer(query.width, query.height, frames)
    return [ImageEmbedding(vector=vector) for vector in vectors]
