from http import HTTPStatus

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", status_code=HTTPStatus.OK)
async def health() -> dict[str, str]:
    return {"status": "ok"}
