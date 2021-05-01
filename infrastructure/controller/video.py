from fastapi import APIRouter
from starlette import status

from application.query import get_video_embedding_query_handler
from application.query.get_video_embedding_query import GetVideoEmbeddingQuery
from infrastructure.controller.model.video_embedding_request import VideoEmbeddingRequest

router = APIRouter()


@router.post("/embedding", status_code=status.HTTP_200_OK)
async def get_embedding(request: VideoEmbeddingRequest):

    return await get_video_embedding_query_handler.handle(GetVideoEmbeddingQuery(request.width,
                                                                                 request.height,
                                                                                 request.frames))
