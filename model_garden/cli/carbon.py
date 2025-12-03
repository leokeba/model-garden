"""Carbon tracking CLI commands for Model Garden.

Provides CLI interface for viewing carbon emissions data, generating reports,
and exporting data in various formats.
"""

import json
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()


@click.group()
def carbon():
    """Carbon emissions tracking and reporting commands."""
    pass


@carbon.command(name="report")
@click.argument("job_id", required=False)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json", "boamps"]),
    default="text",
    help="Output format (text, json, or boamps)",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(),
    help="Output file path (prints to stdout if not specified)",
)
def carbon_report(job_id: str | None, output_format: str, output_file: str | None):
    """View carbon emissions report for a specific job or latest jobs.

    If JOB_ID is not specified, shows the most recent emissions records.

    Examples:
        model-garden carbon report
        model-garden carbon report job_abc123
        model-garden carbon report --format json
        model-garden carbon report job_abc123 --format boamps -o report.json
    """
    from model_garden.carbon import BoAmpsReportGenerator, get_emissions_db

    db = get_emissions_db()

    if job_id:
        # Get specific job
        emission = db.get_emission(job_id)
        if not emission:
            console.print(f"[red]No emissions data found for job: {job_id}[/red]")
            raise SystemExit(1)
        emissions = [emission]
    else:
        # Get recent jobs
        emissions = db.get_all_emissions(limit=10)
        if not emissions:
            console.print("[yellow]No emissions data recorded yet.[/yellow]")
            console.print("\nRun a training job or serve a model to start tracking emissions.")
            return

    if output_format == "json":
        output = json.dumps(emissions, indent=2, default=str)
        if output_file:
            Path(output_file).write_text(output)
            console.print(f"[green]✓ Report saved to {output_file}[/green]")
        else:
            console.print(output)

    elif output_format == "boamps":
        generator = BoAmpsReportGenerator()
        if job_id and len(emissions) == 1:
            report = generator.generate_report(emissions[0])
        else:
            # Generate reports for all
            report = [generator.generate_report(e) for e in emissions]

        output = json.dumps(report, indent=2, default=str)
        if output_file:
            Path(output_file).write_text(output)
            console.print(f"[green]✓ BoAmps report saved to {output_file}[/green]")
        else:
            console.print(output)

    else:
        # Text format
        _print_emissions_text(emissions, job_id is not None)


def _print_emissions_text(emissions: list, single: bool = False):
    """Print emissions data in human-readable format."""
    if single and len(emissions) == 1:
        e = emissions[0]
        _print_single_emission(e)
    else:
        # Summary table for multiple emissions
        table = Table(title="Carbon Emissions Records", show_header=True)
        table.add_column("Job ID", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("CO₂ (kg)", justify="right", style="green")
        table.add_column("Energy (kWh)", justify="right")
        table.add_column("Duration", justify="right")
        table.add_column("Timestamp", style="dim")

        for e in emissions:
            job_id = e.get("job_id", "N/A")[:20]
            job_type = e.get("job_type", "unknown")
            co2 = e.get("emissions_kg_co2", e.get("emissions_kg", 0))
            energy = e.get("energy_consumed_kwh", e.get("energy_consumed", 0))
            duration = e.get("duration_seconds", e.get("duration", 0))
            timestamp = e.get("timestamp", "N/A")

            # Format duration
            if duration < 60:
                dur_str = f"{duration:.0f}s"
            elif duration < 3600:
                dur_str = f"{duration / 60:.1f}m"
            else:
                dur_str = f"{duration / 3600:.1f}h"

            # Format timestamp
            if timestamp and timestamp != "N/A":
                try:
                    ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    ts_str = ts.strftime("%Y-%m-%d %H:%M")
                except (ValueError, AttributeError):
                    ts_str = timestamp[:16]
            else:
                ts_str = "N/A"

            table.add_row(
                job_id,
                job_type,
                f"{co2:.4f}" if co2 else "N/A",
                f"{energy:.4f}" if energy else "N/A",
                dur_str,
                ts_str,
            )

        console.print(table)


def _print_single_emission(e: dict):
    """Print detailed view of a single emission record."""
    job_id = e.get("job_id", "N/A")
    job_type = e.get("job_type", "unknown")

    tree = Tree(f"[bold cyan]Carbon Emissions Report[/bold cyan] - {job_id}")

    # Summary
    summary = tree.add("[bold]Summary[/bold]")
    co2 = e.get("emissions_kg_co2", e.get("emissions_kg", 0))
    energy = e.get("energy_consumed_kwh", e.get("energy_consumed", 0))
    duration = e.get("duration_seconds", e.get("duration", 0))

    summary.add(f"Job Type: [magenta]{job_type}[/magenta]")
    summary.add(f"Total Emissions: [green bold]{co2:.6f} kg CO₂[/green bold]")
    summary.add(f"Energy Consumed: [yellow]{energy:.6f} kWh[/yellow]")

    if duration < 60:
        summary.add(f"Duration: {duration:.1f} seconds")
    elif duration < 3600:
        summary.add(f"Duration: {duration / 60:.1f} minutes")
    else:
        summary.add(f"Duration: {duration / 3600:.2f} hours")

    # Energy breakdown
    if any(
        k in e
        for k in [
            "cpu_energy_kwh",
            "gpu_energy_kwh",
            "ram_energy_kwh",
            "cpu_energy",
            "gpu_energy",
            "ram_energy",
        ]
    ):
        breakdown = tree.add("[bold]Energy Breakdown[/bold]")
        cpu = e.get("cpu_energy_kwh", e.get("cpu_energy", 0))
        gpu = e.get("gpu_energy_kwh", e.get("gpu_energy", 0))
        ram = e.get("ram_energy_kwh", e.get("ram_energy", 0))

        if cpu:
            breakdown.add(f"CPU: {cpu:.6f} kWh")
        if gpu:
            breakdown.add(f"GPU: {gpu:.6f} kWh")
        if ram:
            breakdown.add(f"RAM: {ram:.6f} kWh")

    # Equivalents
    if "equivalents" in e:
        equiv = tree.add("[bold]Environmental Equivalents[/bold]")
        eq = e["equivalents"]
        if "km_driven" in eq:
            equiv.add(f"🚗 {eq['km_driven']:.2f} km driven")
        if "smartphones_charged" in eq:
            equiv.add(f"📱 {int(eq['smartphones_charged'])} smartphones charged")
        if "tree_months" in eq:
            equiv.add(f"🌳 {eq['tree_months']:.1f} tree-months to offset")

    # Metadata
    if e.get("timestamp"):
        meta = tree.add("[bold dim]Metadata[/bold dim]")
        meta.add(f"Timestamp: {e['timestamp']}")
        if e.get("country_iso_code"):
            meta.add(f"Country: {e['country_iso_code']}")
        if e.get("region"):
            meta.add(f"Region: {e['region']}")

    console.print(tree)


@carbon.command(name="summary")
@click.option(
    "--type",
    "-t",
    "job_type",
    type=click.Choice(["all", "training", "inference"]),
    default="all",
    help="Filter by job type",
)
def carbon_summary(job_type: str):
    """Show aggregate carbon emissions summary.

    Displays total emissions, energy consumption, and environmental
    equivalents across all tracked jobs.

    Examples:
        model-garden carbon summary
        model-garden carbon summary --type training
        model-garden carbon summary -t inference
    """
    from model_garden.carbon import get_emissions_db

    db = get_emissions_db()

    # Get totals
    totals = db.get_total_emissions()

    if totals["total_count"] == 0:
        console.print("[yellow]No emissions data recorded yet.[/yellow]")
        console.print("\nRun a training job or serve a model to start tracking emissions.")
        return

    # Create summary panel
    console.print()
    console.print(
        Panel.fit("[bold green]🌱 Carbon Footprint Summary[/bold green]", border_style="green")
    )
    console.print()

    # Overall stats table
    stats = Table(show_header=False, box=None, padding=(0, 2))
    stats.add_column("Metric", style="bold")
    stats.add_column("Value", justify="right")

    stats.add_row("Total Jobs Tracked", f"[cyan]{totals['total_count']}[/cyan]")
    stats.add_row(
        "Total CO₂ Emissions", f"[green bold]{totals['total_emissions_kg_co2']:.6f} kg[/green bold]"
    )
    stats.add_row("Total Energy Used", f"[yellow]{totals['total_energy_kwh']:.6f} kWh[/yellow]")

    # Format duration
    dur = totals["total_duration_seconds"]
    if dur < 60:
        dur_str = f"{dur:.0f} seconds"
    elif dur < 3600:
        dur_str = f"{dur / 60:.1f} minutes"
    else:
        dur_str = f"{dur / 3600:.2f} hours"
    stats.add_row("Total Compute Time", dur_str)

    console.print(stats)
    console.print()

    # By type breakdown
    if totals["by_type"]:
        type_table = Table(title="Breakdown by Job Type", show_header=True)
        type_table.add_column("Type", style="magenta")
        type_table.add_column("Count", justify="right")
        type_table.add_column("CO₂ (kg)", justify="right", style="green")
        type_table.add_column("Energy (kWh)", justify="right")

        for jtype, data in totals["by_type"].items():
            if job_type != "all" and jtype != job_type:
                continue
            type_table.add_row(
                jtype.capitalize(),
                str(data["count"]),
                f"{data['total_co2']:.6f}",
                f"{data['total_energy']:.6f}",
            )

        console.print(type_table)
        console.print()

    # Equivalents
    if totals.get("equivalents"):
        eq = totals["equivalents"]
        equiv_panel = Panel(
            f"🚗 [bold]{eq.get('km_driven', 0):.1f} km[/bold] driven by car\n"
            f"📱 [bold]{int(eq.get('smartphones_charged', 0))}[/bold] smartphone charges\n"
            f"🌳 [bold]{eq.get('tree_months', 0):.1f}[/bold] tree-months needed to offset",
            title="Environmental Equivalents",
            border_style="dim",
        )
        console.print(equiv_panel)


@carbon.command(name="export")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json", "csv", "boamps"]),
    default="json",
    help="Export format",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(),
    required=True,
    help="Output file path",
)
@click.option(
    "--type",
    "-t",
    "job_type",
    type=click.Choice(["all", "training", "inference"]),
    default="all",
    help="Filter by job type",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=None,
    help="Maximum number of records to export",
)
def carbon_export(output_format: str, output_file: str, job_type: str, limit: int | None):
    """Export carbon emissions data to a file.

    Supports JSON, CSV, and BoAmps formats for compliance reporting
    and data analysis.

    Examples:
        model-garden carbon export -o emissions.json
        model-garden carbon export --format csv -o emissions.csv
        model-garden carbon export --format boamps -o report.json --type training
    """
    from model_garden.carbon import BoAmpsReportGenerator, get_emissions_db

    db = get_emissions_db()

    # Get emissions
    filter_type = job_type if job_type != "all" else None
    emissions = db.get_all_emissions(job_type=filter_type, limit=limit)

    if not emissions:
        console.print("[yellow]No emissions data to export.[/yellow]")
        return

    output_path = Path(output_file)

    if output_format == "json":
        output_path.write_text(json.dumps(emissions, indent=2, default=str))
        console.print(f"[green]✓ Exported {len(emissions)} records to {output_file}[/green]")

    elif output_format == "csv":
        import csv

        # Determine all fields
        all_fields = set()
        for e in emissions:
            all_fields.update(e.keys())

        # Remove nested fields
        all_fields.discard("equivalents")
        fields = sorted(all_fields)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(emissions)

        console.print(f"[green]✓ Exported {len(emissions)} records to {output_file}[/green]")

    elif output_format == "boamps":
        generator = BoAmpsReportGenerator()
        reports = [generator.generate_report(e) for e in emissions]

        output_path.write_text(json.dumps(reports, indent=2, default=str))
        console.print(f"[green]✓ Generated {len(reports)} BoAmps reports to {output_file}[/green]")


@carbon.command(name="clear")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Skip confirmation prompt",
)
@click.option(
    "--older-than",
    type=int,
    help="Only clear records older than N days",
)
def carbon_clear(force: bool, older_than: int | None):
    """Clear carbon emissions data.

    Use with caution - this permanently deletes emissions records.

    Examples:
        model-garden carbon clear
        model-garden carbon clear --force
        model-garden carbon clear --older-than 30
    """
    from model_garden.carbon import get_emissions_db

    db = get_emissions_db()
    emissions = db.get_all_emissions()

    if not emissions:
        console.print("[yellow]No emissions data to clear.[/yellow]")
        return

    if older_than:
        # Filter by age
        from datetime import datetime, timedelta

        cutoff = datetime.now(datetime.UTC) - timedelta(days=older_than)
        to_delete = []

        for e in emissions:
            ts = e.get("timestamp")
            if ts:
                try:
                    emission_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if emission_time < cutoff:
                        to_delete.append(e.get("job_id"))
                except (ValueError, AttributeError):
                    pass

        if not to_delete:
            console.print(f"[yellow]No records older than {older_than} days found.[/yellow]")
            return

        if not force:
            if not click.confirm(f"Delete {len(to_delete)} records older than {older_than} days?"):
                console.print("Cancelled.")
                return

        for job_id in to_delete:
            if job_id:
                db.delete_emission(job_id)

        console.print(f"[green]✓ Deleted {len(to_delete)} emission records.[/green]")
    else:
        # Clear all
        if not force:
            if not click.confirm(f"Delete ALL {len(emissions)} emission records?"):
                console.print("Cancelled.")
                return

        for e in emissions:
            job_id = e.get("job_id")
            if job_id:
                db.delete_emission(job_id)

        console.print(f"[green]✓ Cleared all {len(emissions)} emission records.[/green]")
