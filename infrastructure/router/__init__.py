from fastapi import APIRouter

from infrastructure.controller.video import router as video_router

api_router = APIRouter()

api_router.include_router(video_router, prefix="/video", tags=["video"])