# Carbon emissions routes
"""
Routes for carbon emissions tracking:
- GET /api/v1/carbon/emissions - List emissions records
- GET /api/v1/carbon/summary - Get aggregate statistics
- GET /api/v1/carbon/inference/stats - Get inference tracking stats
- GET /api/v1/carbon/boamps/{job_id} - Get BoAmps report
"""

from fastapi import APIRouter, HTTPException, status

from ..storage import get_storage_manager

router = APIRouter(prefix="/api/v1/carbon", tags=["carbon"])


@router.get("/emissions")
async def list_emissions(job_type: str | None = None, limit: int | None = None):
    """List all carbon emissions records."""
    try:
        from model_garden.carbon import get_emissions_db

        storage = get_storage_manager()
        training_jobs = storage.load_training_jobs()

        db = get_emissions_db()
        emissions_records = db.get_all_emissions(job_type=job_type, limit=limit)

        # Format for API
        formatted_emissions = []
        for record in emissions_records:
            job_name = record.get("job_id", "Unknown")
            if record["job_id"] in training_jobs:
                job_name = training_jobs[record["job_id"]].get("name", job_name)

            model_name = record.get("model_name", "Unknown")
            if not model_name or model_name == "Unknown":
                if record["job_id"] in training_jobs:
                    model_name = training_jobs[record["job_id"]].get("base_model", "Unknown")

            formatted_emissions.append(
                {
                    "id": f"emission-{record['job_id']}",
                    "job_id": record["job_id"],
                    "job_name": job_name,
                    "stage": record.get("job_type", "training"),
                    "model_name": model_name,
                    "timestamp": record.get("timestamp", ""),
                    "duration": record.get("duration_seconds", 0.0),
                    "energy_consumed": record.get("energy_consumed_kwh", 0.0),
                    "emissions_kg": record.get("emissions_kg_co2", 0.0),
                    "emissions_rate": record.get("emissions_rate_kg_per_sec", 0.0),
                    "cpu_energy": record.get("cpu_energy_kwh", 0.0),
                    "gpu_energy": record.get("gpu_energy_kwh", 0.0),
                    "ram_energy": record.get("ram_energy_kwh", 0.0),
                    "carbon_intensity": record.get("carbon_intensity_g_per_kwh", 0.0),
                    "country": record.get("country_name", "Unknown"),
                    "region": record.get("region", "Unknown"),
                    "equivalents": record.get("equivalents", {}),
                    "boamps_report": True,
                }
            )

        return {"emissions": formatted_emissions, "count": len(formatted_emissions)}

    except Exception as e:
        import logging

        logging.warning(f"Could not load emissions data: {e}")
        return {"emissions": [], "count": 0}


@router.get("/summary")
async def get_emissions_summary():
    """Get aggregate emissions statistics."""
    try:
        from model_garden.carbon import get_emissions_db

        db = get_emissions_db()
        summary = db.get_total_emissions()
        return summary
    except Exception as e:
        import logging

        logging.warning(f"Could not load emissions summary: {e}")
        return {
            "total_emissions_kg_co2": 0.0,
            "total_energy_kwh": 0.0,
            "total_duration_seconds": 0.0,
            "total_count": 0,
            "by_type": {},
            "equivalents": {},
        }


@router.get("/inference/stats")
async def get_inference_stats():
    """Get current inference carbon tracking statistics."""
    try:
        from model_garden.carbon import get_inference_tracker

        tracker = get_inference_tracker()

        if tracker:
            stats = tracker.get_current_stats()
            return {"tracking": True, **stats}
        else:
            return {
                "tracking": False,
                "message": "No inference tracking active. Load a model to start tracking.",
            }
    except Exception as e:
        return {"tracking": False, "error": str(e)}


@router.get("/boamps/{job_id}")
async def get_boamps_report(job_id: str):
    """Get BoAmps report for a specific job."""
    try:
        from model_garden.carbon import get_boamps_generator, get_emissions_db

        storage = get_storage_manager()
        training_jobs = storage.load_training_jobs()

        db = get_emissions_db()
        emission_data = db.get_emission(job_id)

        if not emission_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No emissions data found for job {job_id}",
            )

        # Get job config if available
        job_config = {}
        if job_id in training_jobs:
            job = training_jobs[job_id]
            job_config = {
                "base_model": job.get("base_model"),
                "dataset_path": job.get("dataset_path"),
                "hyperparameters": job.get("hyperparameters"),
                "lora_config": job.get("lora_config"),
            }

        # Generate report
        generator = get_boamps_generator()
        report = generator.generate_report(
            emissions_data=emission_data, job_config=job_config, report_status="final"
        )

        return report

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating BoAmps report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate BoAmps report: {str(e)}",
        ) from None
