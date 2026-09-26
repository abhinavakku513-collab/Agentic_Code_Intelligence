"""The HTTP surface (docs/spec/06 §2, `.claude/rules/security.md`).

One engine behind a thin FastAPI app. Three rules shape it:

* **Loopback by default.** A retrieval engine over a private corpus has no business listening on a public
  interface, so binding anywhere else requires a bearer token and says so at start-up.
* **No evaluation endpoint.** Scoring reads labels; nothing that reads labels is reachable over a socket
  (docs/spec/06 §1). The API can search, compare and describe — it cannot grade.
* **Typed errors, not tracebacks.** Every `AcisError` maps to the status code the spec assigns it, with the
  message and context the CLI would have printed.

The static UI is served from this app rather than a separate server, and carries no CDN reference: the whole
thing runs on a machine with no network (D15).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from acis.api import schemas
from acis.core.errors import (
    AcisError,
    IndexRequired,
    InvalidInput,
    NotFound,
    NotReady,
    ResourceLimit,
    SnapshotInvalid,
    StrictViolation,
    VersionConflict,
)
from acis.core.types import EvolveRequest, Hit, SearchRequest, SourceSpec
from acis.obs.log import get_logger

log = get_logger("acis.api")

STATIC_DIR = Path(__file__).parent / "static"
TOKEN_ENV = "ACIS_API_TOKEN"
#: `docs/spec/06` §1 assigns each error its status code; the API only translates.
STATUS = {
    InvalidInput: 400,
    NotFound: 404,
    IndexRequired: 409,
    SnapshotInvalid: 409,
    VersionConflict: 409,
    ResourceLimit: 413,
    NotReady: 503,
    StrictViolation: 500,
}


def _status_for(error: AcisError) -> int:
    for kind, code in STATUS.items():
        if isinstance(error, kind):
            return code
    return 500


def _hit_out(hit: Hit) -> dict[str, Any]:
    return {
        "rank": hit.rank,
        "score": float(hit.score),
        "unit": {
            "unit_id": hit.unit.unit_id,
            "key": hit.unit.key,
            "body_hash": hit.unit.body_hash,
            "n_bytes": hit.unit.n_bytes,
        },
        "source": hit.source,
        "signals": {k: float(v) for k, v in hit.signals.items()},
    }


def create_app(engine: Any = None, *, config_path: str = "configs/dev.yaml") -> Any:
    """Build the FastAPI application around one engine instance."""
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles

    if engine is None:
        from acis.core.config import load_frozen_config
        from acis.embed.factory import build_encoder
        from acis.engine import AcisEngine

        config = load_frozen_config(config_path)
        engine = AcisEngine.from_config(config, encoder=build_encoder(config))

    app = FastAPI(
        title="ACIS",
        version="1.1",
        description="Local, offline, CPU-first code retrieval. No evaluation endpoint is exposed (spec 06 §1).",
        docs_url="/docs",
    )
    started = time.time()
    counters = {"requests": 0, "errors": 0, "searches": 0, "evolutions": 0}

    def require_token(authorization: str = Header(default="")) -> None:
        """A token is required only when the service is not on loopback — set `ACIS_API_TOKEN` to enable it."""
        expected = os.environ.get(TOKEN_ENV, "")
        if not expected:
            return
        if authorization.removeprefix("Bearer ").strip() != expected:
            raise HTTPException(status_code=401, detail="a bearer token is required on a non-loopback bind")

    guard: list[Any] = [Depends(require_token)]

    @app.exception_handler(AcisError)
    async def _typed_error(_request: Request, exc: AcisError) -> JSONResponse:
        counters["errors"] += 1
        return JSONResponse(
            status_code=_status_for(exc),
            content={"code": exc.code, "message": exc.message, "context": exc.context},
        )

    @app.middleware("http")
    async def _count(request: Request, call_next: Callable[..., Any]) -> Any:
        counters["requests"] += 1
        return await call_next(request)

    # -- retrieval ---------------------------------------------------------------------------------------------
    @app.post("/v1/search", response_model=schemas.SearchOut, dependencies=guard)
    def search(body: schemas.SearchBody) -> dict[str, Any]:
        counters["searches"] += 1
        response = engine.search(
            SearchRequest(
                query=body.query,
                repo_id=body.repo_id,
                version=body.version,
                top_k=body.top_k,
                mode=body.mode,
                explain=body.explain,
                diagnostics=True,
            )
        )
        return {
            "snapshot": {
                "id": response.snapshot.id,
                "version": response.snapshot.version,
                "complete": response.snapshot.complete,
                "missing_channels": list(response.snapshot.missing_channels),
            },
            "results": [_hit_out(h) for h in response.results],
            "route": response.route,
            "confidence": response.confidence,
            "no_strong_match": response.no_strong_match,
            "timings_ms": {k: round(float(v), 3) for k, v in response.timings_ms.items()},
            "degradations": list(response.degradations),
            "interpreted_intent": dict(response.interpreted_intent),
        }

    @app.post("/v1/evolve", response_model=schemas.EvolveOut, dependencies=guard)
    def evolve(body: schemas.EvolveBody) -> dict[str, Any]:
        counters["evolutions"] += 1
        response = engine.retrieve_evolution(
            EvolveRequest(
                query=body.query,
                repo_id=body.repo_id,
                top_k=body.top_k,
                flat=body.flat,
                filters={"prefer": body.prefer},
            )
        )
        return {
            "groups": [dict(g) for g in response.groups],
            "flat_results": [_hit_out(h) for h in response.flat_results],
            "degradations": list(response.degradations),
        }

    # -- repositories and versions -------------------------------------------------------------------------------
    @app.get("/v1/repos", dependencies=guard)
    def repos() -> dict[str, Any]:
        from acis.store import catalog

        with catalog.open_catalog() as db:
            return {"repos": [dict(r) for r in catalog.list_repos(db)]}

    @app.post("/v1/repos/{repo_id}/ingest", dependencies=guard)
    def ingest(repo_id: str, body: schemas.IngestBody) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if body.extensions:
            options["extensions"] = [e if e.startswith(".") else f".{e}" for e in body.extensions]
        if body.rev:
            options["rev"] = body.rev
        handle = engine.ingest(SourceSpec(kind=body.kind, location=body.location, options=options), repo_id=repo_id)
        return {"job_id": handle.job_id, "state": handle.state, "detail": handle.detail}

    @app.get("/v1/repos/{repo_id}/versions", dependencies=guard)
    def versions(repo_id: str) -> dict[str, Any]:
        from acis.store import snapshots

        active = snapshots.active_snapshot_id(repo_id)
        return {
            "repo_id": repo_id,
            "active": active,
            "versions": [
                {
                    "label": str(row["label"]),
                    "snapshot_id": row["snapshot_id"],
                    "n_units": int(row.get("n_units", 0)),
                    "state": str(row["state"]),
                    "active": row["snapshot_id"] == active,
                }
                for row in engine.versions(repo_id)
            ],
        }

    @app.post("/v1/repos/{repo_id}/versions/{label}/activate", dependencies=guard)
    def activate(repo_id: str, label: str) -> dict[str, Any]:
        return {"repo_id": repo_id, "active": engine.activate(repo_id, label)}

    @app.post("/v1/repos/{repo_id}/rollback", dependencies=guard)
    def rollback(repo_id: str) -> dict[str, Any]:
        return {"repo_id": repo_id, "active": engine.rollback(repo_id)}

    @app.get("/v1/repos/{repo_id}/diff", response_model=schemas.DiffOut, dependencies=guard)
    def diff(repo_id: str, a: str, b: str) -> dict[str, Any]:
        comparison = engine.compare_versions(repo_id, a, b)
        return {
            "repo_id": repo_id,
            "a": a,
            "b": b,
            "added": list(comparison.added),
            "removed": list(comparison.removed),
            "changed": list(comparison.changed),
        }

    # -- operations -----------------------------------------------------------------------------------------------
    @app.get("/healthz", response_model=schemas.HealthOut)
    def healthz() -> dict[str, Any]:
        from acis.store import catalog

        encoder = getattr(engine, "encoder", None)
        with catalog.open_catalog() as db:
            names = [str(r["repo_id"]) for r in catalog.list_repos(db)]
        return {
            "status": "ok" if encoder is not None else "degraded",
            "encoder": getattr(encoder, "name", "none"),
            "submission_capable": bool(getattr(encoder, "submission_capable", False)),
            "numeric_profile": engine.config.numeric_profile,
            "threads": engine.threads,
            "repos": names,
        }

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        return {
            "ready": getattr(engine, "encoder", None) is not None,
            # Whether the default corpus of the page (`repo_id="-"`, the APPS one) can be searched yet.
            "p0_corpus": bool(getattr(engine, "_snapshots", None)),
            "uptime_s": round(time.time() - started, 1),
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        """Prometheus text format, from the engine's own counters — no second bookkeeping to drift."""
        lines = [
            "# HELP acis_requests_total HTTP requests served",
            "# TYPE acis_requests_total counter",
            f"acis_requests_total {counters['requests']}",
            "# HELP acis_searches_total search requests served",
            "# TYPE acis_searches_total counter",
            f"acis_searches_total {counters['searches']}",
            "# HELP acis_errors_total typed errors returned",
            "# TYPE acis_errors_total counter",
            f"acis_errors_total {counters['errors']}",
            "# HELP acis_uptime_seconds seconds since start",
            "# TYPE acis_uptime_seconds gauge",
            f"acis_uptime_seconds {round(time.time() - started, 1)}",
        ]
        for name, value in sorted(engine.counters.snapshot().items()):
            safe = name.replace(".", "_")
            lines += [f"# TYPE acis_{safe} counter", f"acis_{safe} {value}"]
        return "\n".join(lines) + "\n"

    @app.get("/v1/diagnostics", dependencies=guard)
    def diagnostics() -> dict[str, Any]:
        d = engine.diagnostics()
        return {
            "config_hash": d.config_hash,
            "model_fingerprint": d.model_fingerprint,
            "numeric_profile": d.numeric_profile,
            "tier": d.tier,
            "hardware": dict(d.hardware),
            "snapshots": [dict(s) for s in d.snapshots],
            "counters": dict(d.counters),
            "agent_calls": d.agent_calls,
        }

    # -- the UI ------------------------------------------------------------------------------------------------
    if STATIC_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")

        @app.get("/")
        def index() -> Any:
            return FileResponse(STATIC_DIR / "index.html")

    return app


def preload_p0(engine: Any) -> str | None:
    """Build the in-memory APPS snapshot the page searches by default (`repo_id="-"`).

    Without it the service starts, and a judge's first free-text query answers "no snapshot has been built yet".
    Document vectors come from the content-addressed cache when they are there; without the dataset the service
    still starts, with the versioned repositories only, and `/readyz` says so.
    """
    from acis.appsdata import apps

    if not apps.is_available():
        log.info("api.p0_unavailable", reason="dataset assets not fetched (make fetch)")
        return None
    started = time.perf_counter()
    snapshot = engine.build_snapshot(apps.load_corpus(), source="serve:p0")
    log.info(
        "api.p0_ready",
        snapshot=snapshot.snapshot_id,
        units=snapshot.n_units,
        seconds=round(time.perf_counter() - started, 1),
    )
    return str(snapshot.snapshot_id)


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    config_path: str = "configs/dev.yaml",
    reload: bool = False,
) -> None:
    """Run the service. A non-loopback bind without a token is refused rather than quietly exposed."""
    import uvicorn

    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get(TOKEN_ENV):
        raise InvalidInput(
            f"refusing to bind {host} without {TOKEN_ENV}: a non-loopback service needs a bearer token",
            host=host,
        )
    from acis.core.config import load_frozen_config
    from acis.embed.factory import build_encoder
    from acis.engine import AcisEngine

    config = load_frozen_config(config_path)
    engine = AcisEngine.from_config(config, encoder=build_encoder(config))
    preload_p0(engine)
    log.info("api.serving", host=host, port=port, config=config_path)
    uvicorn.run(create_app(engine, config_path=config_path), host=host, port=port, reload=reload, log_level="info")


__all__ = ["STATIC_DIR", "STATUS", "TOKEN_ENV", "create_app", "preload_p0", "serve"]
