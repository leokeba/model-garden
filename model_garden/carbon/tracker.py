"""Carbon emissions tracker wrapper around CodeCarbon."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codecarbon import EmissionsTracker
from rich.console import Console

from .database import get_emissions_db

console = Console()


class CarbonTracker:
    """
    Wrapper around CodeCarbon's EmissionsTracker for Model Garden.

    Provides a simplified interface for tracking carbon emissions during
    training and inference operations.
    """

    def __init__(
        self,
        job_id: str,
        job_type: str = "training",  # "training" or "inference"
        output_dir: Path | None = None,
        project_name: str | None = None,
        country_iso_code: str = "USA",
    ):
        """
        Initialize carbon tracker.

        Args:
            job_id: Unique identifier for the job
            job_type: Type of job ("training" or "inference")
            output_dir: Directory to save emissions data (default: storage/logs/{job_id})
            project_name: Name for the CodeCarbon project
            country_iso_code: ISO code for carbon intensity (default: USA)
        """
        self.job_id = job_id
        self.job_type = job_type

        # Set output directory
        if output_dir is None:
            output_dir = Path(f"storage/logs/{job_id}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Project name
        if project_name is None:
            project_name = f"{job_type}-{job_id}"

        # Initialize CodeCarbon tracker with optimized settings
        # Use 'machine' mode to capture ALL GPU/CPU activity including vLLM processes
        self.tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=str(self.output_dir),
            output_file="emissions.csv",
            log_level="error",  # Suppress verbose logs
            save_to_file=True,
            save_to_api=False,
            tracking_mode="machine",  # Track entire machine to capture vLLM GPU usage
            measure_power_secs=1,  # Sample every 1 second for faster inference tracking
        )

        self.emissions_data: dict[str, Any] | None = None
        self.started = False

    def start(self) -> None:
        """Start tracking emissions."""
        if self.started:
            console.print("[yellow]⚠️  Carbon tracker already started[/yellow]")
            return

        try:
            self.tracker.start()
            self.started = True
            console.print(f"[green]🌍 Carbon tracking started for {self.job_id}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to start carbon tracking: {e}[/yellow]")
            console.print("[yellow]Continuing without carbon tracking...[/yellow]")

    def get_live_emissions(self) -> dict[str, Any] | None:
        """
        Get current emissions without stopping the tracker.

        Returns:
            Dictionary with emissions and energy data, or None if unavailable
        """
        if not self.started:
            return None

        try:
            # Flush current measurements and get emissions data
            self.tracker.flush()

            # Use CodeCarbon's _prepare_emissions_data to get live data
            if hasattr(self.tracker, "_prepare_emissions_data"):
                emissions_data = self.tracker._prepare_emissions_data()
                if emissions_data:
                    return {
                        "emissions_kg_co2": getattr(emissions_data, "emissions", 0.0),
                        "energy_consumed_kwh": getattr(emissions_data, "energy_consumed", 0.0),
                        "cpu_energy_kwh": getattr(emissions_data, "cpu_energy", 0.0),
                        "gpu_energy_kwh": getattr(emissions_data, "gpu_energy", 0.0),
                        "ram_energy_kwh": getattr(emissions_data, "ram_energy", 0.0),
                        "duration_seconds": getattr(emissions_data, "duration", 0.0),
                        "cpu_power_watts": getattr(emissions_data, "cpu_power", 0.0),
                        "gpu_power_watts": getattr(emissions_data, "gpu_power", 0.0),
                        "ram_power_watts": getattr(emissions_data, "ram_power", 0.0),
                    }

            # Fallback: try _emissions attribute
            if hasattr(self.tracker, "_emissions"):
                emissions_obj = self.tracker._emissions
                if hasattr(emissions_obj, "values"):
                    values: dict = emissions_obj.values  # type: ignore[assignment]
                    return {
                        "emissions_kg_co2": values.get("emissions", 0.0),
                        "energy_consumed_kwh": values.get("energy_consumed", 0.0),
                        "cpu_energy_kwh": values.get("cpu_energy", 0.0),
                        "gpu_energy_kwh": values.get("gpu_energy", 0.0),
                        "ram_energy_kwh": values.get("ram_energy", 0.0),
                        "duration_seconds": values.get("duration", 0.0),
                        "cpu_power_watts": values.get("cpu_power", 0.0),
                        "gpu_power_watts": values.get("gpu_power", 0.0),
                        "ram_power_watts": values.get("ram_power", 0.0),
                    }

            return None
        except Exception as e:
            console.print(f"[dim]Could not get live emissions: {e}[/dim]")
            return None

    def stop(self) -> dict[str, Any] | None:
        """
        Stop tracking and save emissions data.

        Returns:
            Dictionary with emissions data, or None if tracking failed
        """
        if not self.started:
            console.print("[yellow]⚠️  Carbon tracker not started[/yellow]")
            return None

        try:
            # Stop tracker and get emissions
            emissions_kg = self.tracker.stop()

            # Get final emissions data
            if emissions_kg is not None:
                self.emissions_data = self._get_emissions_summary(emissions_kg)
            else:
                self.emissions_data = self._get_emissions_summary(0.0)

            # Save to JSON for easy API access
            json_path = self.output_dir / "emissions.json"
            with open(json_path, "w") as f:
                json.dump(self.emissions_data, f, indent=2)

            # Save to persistent database
            try:
                db = get_emissions_db()
                db.add_emission(self.emissions_data)
            except Exception as e:
                console.print(f"[yellow]⚠️  Failed to save to emissions database: {e}[/yellow]")

            console.print(
                f"[green]✅ Carbon tracking complete: "
                f"{self.emissions_data['emissions_kg_co2']:.6f} kg CO2[/green]"
            )
            console.print(f"[dim]Emissions data saved to {json_path}[/dim]")

            return self.emissions_data

        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to stop carbon tracking: {e}[/yellow]")
            return None

    def _get_emissions_summary(self, emissions_kg: float) -> dict[str, Any]:
        """
        Create a comprehensive emissions summary.

        Args:
            emissions_kg: Total emissions in kg CO2

        Returns:
            Dictionary with detailed emissions data
        """
        # Try to read the CSV file for detailed data
        csv_path = self.output_dir / "emissions.csv"
        detailed_data = {}

        if csv_path.exists():
            try:
                import csv

                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    # Get the last row (most recent measurement)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]
                        energy_consumed = float(last_row.get("energy_consumed", 0))
                        emissions_from_csv = float(last_row.get("emissions", 0))

                        # Calculate carbon intensity from emissions and energy
                        # Formula: carbon_intensity (g/kWh) = emissions (kg) * 1000 / energy (kWh)
                        carbon_intensity = 0.0
                        if energy_consumed > 0 and emissions_from_csv > 0:
                            carbon_intensity = (emissions_from_csv * 1000) / energy_consumed

                        detailed_data = {
                            "energy_consumed_kwh": energy_consumed,
                            "duration_seconds": float(last_row.get("duration", 0)),
                            "emissions_rate_kg_per_sec": float(last_row.get("emissions_rate", 0)),
                            "cpu_power_watts": float(last_row.get("cpu_power", 0)),
                            "gpu_power_watts": float(last_row.get("gpu_power", 0)),
                            "ram_power_watts": float(last_row.get("ram_power", 0)),
                            "cpu_energy_kwh": float(last_row.get("cpu_energy", 0)),
                            "gpu_energy_kwh": float(last_row.get("gpu_energy", 0)),
                            "ram_energy_kwh": float(last_row.get("ram_energy", 0)),
                            "carbon_intensity_g_per_kwh": carbon_intensity,
                            "country_name": last_row.get("country_name", "Unknown"),
                            "region": last_row.get("region", "Unknown"),
                        }
            except Exception as e:
                console.print(f"[dim]Could not parse emissions CSV: {e}[/dim]")

        # Build summary
        summary = {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "timestamp": datetime.now(UTC).isoformat(),
            "emissions_kg_co2": emissions_kg,
            "tracking_mode": "process",
            "output_dir": str(self.output_dir),
        }

        # Add detailed data if available
        summary.update(detailed_data)

        # Calculate equivalent metrics for better understanding
        if emissions_kg > 0:
            # Equivalents (approximate)
            summary["equivalents"] = {
                "km_driven": emissions_kg * 4.6,  # km in average car
                "smartphones_charged": int(emissions_kg * 121),  # full charges
                "tree_months": emissions_kg / 0.006,  # months of tree absorption
            }

        return summary

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def get_emissions_summary(
    job_id: str, logs_dir: Path = Path("storage/logs")
) -> dict[str, Any] | None:
    """
    Get emissions summary for a completed job.

    Args:
        job_id: Job identifier
        logs_dir: Base directory for logs

    Returns:
        Emissions data dictionary, or None if not found
    """
    emissions_file = logs_dir / job_id / "emissions.json"

    if not emissions_file.exists():
        # Try to read from CSV as fallback
        csv_file = logs_dir / job_id / "emissions.csv"
        if csv_file.exists():
            console.print(f"[dim]Found emissions.csv for {job_id}, parsing...[/dim]")
            # Parse CSV and create JSON
            try:
                import csv

                with open(csv_file) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]
                        return {
                            "job_id": job_id,
                            "emissions_kg_co2": float(last_row.get("emissions", 0)),
                            "energy_consumed_kwh": float(last_row.get("energy_consumed", 0)),
                            "duration_seconds": float(last_row.get("duration", 0)),
                        }
            except Exception as e:
                console.print(f"[yellow]Failed to parse emissions CSV: {e}[/yellow]")
        return None

    try:
        with open(emissions_file) as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[yellow]Failed to read emissions data: {e}[/yellow]")
        return None


def list_all_emissions(logs_dir: Path = Path("storage/logs")) -> list[dict[str, Any]]:
    """
    List all emissions records from completed jobs.

    Args:
        logs_dir: Base directory for logs

    Returns:
        List of emissions data dictionaries
    """
    emissions_records = []

    if not logs_dir.exists():
        return emissions_records

    # Scan all job directories
    for job_dir in logs_dir.iterdir():
        if job_dir.is_dir():
            emissions_data = get_emissions_summary(job_dir.name, logs_dir)
            if emissions_data:
                emissions_records.append(emissions_data)

    # Sort by timestamp (newest first)
    emissions_records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return emissions_records
