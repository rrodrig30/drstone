"""Standalone Dr Stone web app.

Serves the point-of-care kidney-stone decision-support tool (composition
distribution + acute management + prevention) behind a login. Run:

    uvicorn web.app:app --host 127.0.0.1 --port 8200
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from drstone import auth
from drstone.pages import router as pages_router
from drstone.web import router as app_router

app = FastAPI(title="Dr Stone — Kidney-Stone Decision Support")

_HERE = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")
app.include_router(pages_router)
app.include_router(app_router)


@app.on_event("startup")
def _startup() -> None:
    auth.seed_superuser()
