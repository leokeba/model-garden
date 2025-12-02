# FastAPI application factory
"""
Creates and configures the FastAPI application with:
- Lifespan management (startup/shutdown)
- CORS middleware
- Router registration
- Static file serving for frontend
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import (
    carbon_router,
    datasets_router,
    inference_router,
    models_router,
    system_router,
    training_router,
)
from .storage import get_storage_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    print("🚀 Starting Model Garden API...")

    # Initialize storage
    storage = get_storage_manager()
    training_jobs = storage.load_training_jobs()

    # Mark any running jobs as interrupted (crashed during previous run)
    for _job_id, job in training_jobs.items():
        if job["status"] == "running":
            job["status"] = "failed"
            job["error_message"] = "Job interrupted by server restart"
    storage.save_training_jobs(training_jobs)

    # Start the queue worker for autonomous job processing
    try:
        from model_garden.queue import get_queue_worker

        worker = get_queue_worker()
        await worker.start()
        print("✓ Queue worker started")
    except Exception as e:
        print(f"⚠️  Failed to start queue worker: {e}")

    print("✓ Model Garden API ready")

    yield

    # Shutdown
    print("🛑 Shutting down Model Garden API...")

    # Stop inference carbon tracking if active
    try:
        from model_garden.carbon import stop_inference_tracker

        emissions_data = stop_inference_tracker()
        if emissions_data:
            print(
                f"✓ Final inference emissions saved: {emissions_data['emissions_kg_co2']:.6f} kg CO2"
            )
    except Exception as e:
        print(f"⚠️  Failed to stop inference tracker: {e}")

    # Stop the queue worker
    try:
        from model_garden.queue import get_queue_worker

        worker = get_queue_worker()
        await worker.stop()
        print("✓ Queue worker stopped")
    except Exception as e:
        print(f"⚠️  Failed to stop queue worker: {e}")

    print("✓ Model Garden API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Model Garden API",
        description="API for fine-tuning and serving LLMs/VLMs with carbon tracking",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(models_router)
    app.include_router(training_router)
    app.include_router(inference_router)
    app.include_router(datasets_router)
    app.include_router(carbon_router)
    app.include_router(system_router)

    # Mount frontend static files if available
    frontend_build_path = Path(__file__).parent.parent.parent / "frontend" / "build"
    if frontend_build_path.exists():
        # Serve static assets
        app.mount(
            "/_app", StaticFiles(directory=str(frontend_build_path / "_app")), name="static-assets"
        )

        # Catch-all route for SvelteKit client-side routing
        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str):
            """Serve the SvelteKit SPA for all non-API routes."""
            # Try specific HTML files first
            html_file = frontend_build_path / f"{full_path}.html"
            if html_file.exists():
                return FileResponse(html_file)

            # Check directory with index.html
            dir_path = frontend_build_path / full_path
            if dir_path.is_dir():
                index_file = dir_path / "index.html"
                if index_file.exists():
                    return FileResponse(index_file)

            # Fallback to main index.html
            return FileResponse(frontend_build_path / "index.html")
    else:
        print(
            "⚠️  Frontend build not found. Run 'cd frontend && npm run build' to build the frontend."
        )

    return app


# Create the default app instance
app = create_app()
