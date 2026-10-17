"""HTTP error contract: every error body is `{"error": {"code", "message"}}`.

`AppError`s (project errors, defined in shared_kernel or by modules) are mapped by `kind`, so
this module knows no business type.
"""

from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from agintis.shared_kernel.errors import AppError, ErrorKind

STATUS_BY_KIND: dict[ErrorKind, HTTPStatus] = {
    ErrorKind.NOT_FOUND: HTTPStatus.NOT_FOUND,
    ErrorKind.INVALID: HTTPStatus.UNPROCESSABLE_ENTITY,
    ErrorKind.BAD_REQUEST: HTTPStatus.BAD_REQUEST,
    ErrorKind.CONFLICT: HTTPStatus.CONFLICT,
    ErrorKind.UNAVAILABLE: HTTPStatus.SERVICE_UNAVAILABLE,
    ErrorKind.UNAUTHORIZED: HTTPStatus.UNAUTHORIZED,
    ErrorKind.FORBIDDEN: HTTPStatus.FORBIDDEN,
}

INVALID_REQUEST_CODE = "invalid_request"
MAX_VALIDATION_DETAILS = 5


def error_body(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


async def _handle_app_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    status = STATUS_BY_KIND[exc.kind]
    return JSONResponse(
        error_body(exc.code, exc.message), status_code=status, headers=dict(exc.headers)
    )


async def _handle_validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Location + rule only: the raw input is never echoed back.
    details = "; ".join(
        f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
        for err in exc.errors()[:MAX_VALIDATION_DETAILS]
    )
    return JSONResponse(
        error_body(INVALID_REQUEST_CODE, details or "Invalid request"),
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
