import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agintis.bootstrap.container import Container
from agintis.bootstrap.settings import Settings, get_settings
from agintis.modules.agents_catalog.http import router as agents_router
from agintis.modules.conversations.http import router as threads_router
from agintis.platform.http.errors import register_error_handlers
from agintis.platform.http.health import router as health_router


def create_app(settings: Settings | None = None) -> FastAPI:
    """App factory. With no args (``uvicorn --factory``) settings come from env/.env."""
    resolved = settings if settings is not None else get_settings()
    logging.basicConfig(level=resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with Container.build(resolved, with_tenant_resolver=True) as container:
            app.state.container = container
            app.state.tenant_resolver = container.require_tenant_resolver()
            app.state.agents_catalog = container.agents_catalog
            app.state.threads = container.threads
            yield

    app = FastAPI(title="agintis", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(agents_router)
    app.include_router(threads_router)
    register_error_handlers(app)
    return app
