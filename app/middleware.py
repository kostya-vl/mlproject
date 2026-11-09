"""
Middleware: обрывает запрос, как только тело превысило MAX_BODY_BYTES.

Content-Length — быстрая отсечка, затем потоковое чтение: как только
суммарно прочитано > лимита, возвращаем 413 и не читаем остальное.
Прочитанное тело кэшируется в request._body — downstream (FastAPI)
читает его через request.body() без повторного чтения стрима.
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        # 1. Быстрая отсечка по заголовку (не доверяем, но экономим чтение)
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > self.max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "request body too large",
                                 "limit_bytes": self.max_bytes},
                    )
            except ValueError:
                return JSONResponse(status_code=400,
                                    content={"detail": "invalid Content-Length"})

        # 2. Потоковое чтение с лимитом — не даём загрузить больше в память
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > self.max_bytes:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "request body too large",
                             "limit_bytes": self.max_bytes},
                )

        # 3. Кэшируем тело в request._body — body() вернёт его без повторного
        #    чтения стрима. Передаём в call_next ТОТ ЖЕ request (не новый),
        #    иначе downstream получит пустое тело.
        request._body = bytes(body)
        return await call_next(request)
