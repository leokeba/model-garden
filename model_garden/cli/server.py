"""Server commands for Model Garden CLI.

Contains:
- serve: Start the FastAPI server (main API with UI)
"""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind the server to",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to bind the server to",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development",
)
def serve(host: str, port: int, reload: bool) -> None:
    """Start the FastAPI server.

    Example:

        \b
        # Start server on default host/port
        uv run model-garden serve

        \b
        # Start with auto-reload for development
        uv run model-garden serve --reload

        \b
        # Start on custom host/port
        uv run model-garden serve --host 127.0.0.1 --port 3000
    """
    try:
        import uvicorn

        console.print("\n[bold cyan]🌱 Model Garden - API Server[/bold cyan]\n")
        console.print(f"[cyan]Starting server on http://{host}:{port}[/cyan]")

        if reload:
            console.print("[yellow]⚠️  Auto-reload enabled (development mode)[/yellow]")

        console.print("[green]✓[/green] Server starting...\n")

        uvicorn.run(
            "model_garden.api:app",
            host=host,
            port=port,
            reload=reload,
        )

    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        raise click.Abort()
