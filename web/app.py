"""Standalone Dr Stone web app.

Serves the point-of-care uric-acid stone predictor (calibrated probability from a
non-contrast CT + routine labs). Run:

    uvicorn web.app:app --host 127.0.0.1 --port 8200
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from drstone.web import router

app = FastAPI(title="Dr Stone — Uric-Acid Stone Predictor")

_HERE = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")
app.include_router(router)


@app.get("/")
def root():
    return RedirectResponse("/drstone")
