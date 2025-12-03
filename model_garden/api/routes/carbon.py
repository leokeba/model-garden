# Carbon emissions routes
"""
Routes for carbon emissions tracking:
- GET /api/v1/carbon/emissions - List emissions records
- GET /api/v1/carbon/summary - Get aggregate statistics
- GET /api/v1/carbon/inference/stats - Get inference tracking stats
- GET /api/v1/carbon/boamps/{job_id} - Get BoAmps report
- GET /api/v1/carbon/analytics/trends - Get emissions trends over time
- GET /api/v1/carbon/analytics/comparisons - Compare models/jobs
- GET /api/v1/carbon/analytics/recommendations - Get carbon reduction tips
"""

from collections import defaultdict
from datetime import UTC, datetime, timedelta

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

            model_name = record.get("model_name") or "Unknown"
            base_model = record.get("base_model")

            # Fallback: if model_name is Unknown, try to get from training jobs
            if model_name == "Unknown" and record["job_id"] in training_jobs:
                job = training_jobs[record["job_id"]]
                model_name = job.get("name") or job.get("base_model") or model_name

            # Use base_model from record, or fallback to training job
            if not base_model:
                if record["job_id"] in training_jobs:
                    base_model = training_jobs[record["job_id"]].get("base_model")

            # Calculate carbon intensity if it's 0 (for historical data)
            carbon_intensity = record.get("carbon_intensity_g_per_kwh", 0.0)
            if carbon_intensity == 0.0:
                emissions_kg = record.get("emissions_kg_co2", 0.0)
                energy_kwh = record.get("energy_consumed_kwh", 0.0)
                if energy_kwh > 0 and emissions_kg > 0:
                    # carbon_intensity (g/kWh) = emissions (kg) * 1000 / energy (kWh)
                    carbon_intensity = (emissions_kg * 1000) / energy_kwh

            formatted_emissions.append(
                {
                    "id": f"emission-{record['job_id']}",
                    "job_id": record["job_id"],
                    "job_name": job_name,
                    "stage": record.get("job_type", "training"),
                    "model_name": model_name,
                    "base_model": base_model,
                    "timestamp": record.get("timestamp", ""),
                    "duration": record.get("duration_seconds", 0.0),
                    "energy_consumed": record.get("energy_consumed_kwh", 0.0),
                    "emissions_kg": record.get("emissions_kg_co2", 0.0),
                    "emissions_rate": record.get("emissions_rate_kg_per_sec", 0.0),
                    "cpu_energy": record.get("cpu_energy_kwh", 0.0),
                    "gpu_energy": record.get("gpu_energy_kwh", 0.0),
                    "ram_energy": record.get("ram_energy_kwh", 0.0),
                    "carbon_intensity": carbon_intensity,
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
    """Get BoAmps report for a specific job.

    Generates a BoAmps v1.1.0 compliant report containing:
    - Task information (stage, family, algorithms, datasets)
    - Energy measurements (power consumption, duration)
    - Infrastructure details (GPU, CPU, RAM)
    - System and software environment
    - Carbon intensity and location data
    """
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

        # Build comprehensive job config for BoAmps report
        job_config = {}
        if job_id in training_jobs:
            job = training_jobs[job_id]
            job_config = {
                # Core model info
                "base_model": job.get("base_model"),
                "model_type": job.get("model_type"),
                "is_vision": job.get("is_vision", False),
                # Dataset info
                "dataset_path": job.get("dataset_path"),
                "from_hub": job.get("from_hub", False),
                "validation_dataset_path": job.get("validation_dataset_path"),
                "validation_from_hub": job.get("validation_from_hub", False),
                # Training config
                "hyperparameters": job.get("hyperparameters", {}),
                "lora_config": job.get("lora_config"),
                "selective_loss": job.get("selective_loss", False),
                "max_seq_length": job.get("max_seq_length"),
                "save_method": job.get("save_method"),
                # Progress/metrics
                "current_step": job.get("current_step"),
                "total_steps": job.get("total_steps"),
                "current_epoch": job.get("current_epoch"),
            }

            # Extract dataset size from metrics if available
            metrics = job.get("metrics", {})
            if metrics:
                # Try to get dataset info from training metrics
                training_metrics = metrics.get("training", [])
                if training_metrics and len(training_metrics) > 0:
                    # Estimate samples from steps and batch size
                    hyperparams = job.get("hyperparameters", {})
                    batch_size = hyperparams.get("batch_size", 1)
                    grad_accum = hyperparams.get("gradient_accumulation_steps", 1)
                    total_steps = job.get("total_steps", 0)
                    epochs = hyperparams.get("num_epochs", 1)
                    if total_steps > 0 and epochs > 0:
                        # samples = steps * batch_size * grad_accum / epochs
                        estimated_samples = int((total_steps * batch_size * grad_accum) / epochs)
                        if estimated_samples > 0:
                            job_config["dataset_num_samples"] = estimated_samples

            # Get final loss if available
            if metrics.get("training"):
                last_metric = metrics["training"][-1]
                if "loss" in last_metric:
                    job_config["final_loss"] = last_metric["loss"]

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


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================


@router.get("/analytics/trends")
async def get_emissions_trends(
    period: str = "7d",  # 7d, 30d, 90d, all
    granularity: str = "day",  # hour, day, week, month
):
    """
    Get emissions trends over time.

    Returns time-series data for charting emissions and energy consumption.
    """
    try:
        from model_garden.carbon import get_emissions_db

        db = get_emissions_db()
        all_emissions = db.get_all_emissions()

        if not all_emissions:
            return {
                "period": period,
                "granularity": granularity,
                "data_points": [],
                "totals": {
                    "emissions_kg": 0,
                    "energy_kwh": 0,
                    "job_count": 0,
                },
            }

        # Calculate cutoff date
        now = datetime.now(UTC)
        if period == "7d":
            cutoff = now - timedelta(days=7)
        elif period == "30d":
            cutoff = now - timedelta(days=30)
        elif period == "90d":
            cutoff = now - timedelta(days=90)
        else:  # all
            cutoff = datetime.min

        # Filter and group by time period
        grouped_data: dict[str, dict] = defaultdict(
            lambda: {
                "emissions_kg": 0.0,
                "energy_kwh": 0.0,
                "job_count": 0,
                "training": 0,
                "inference": 0,
            }
        )

        for record in all_emissions:
            ts_str = record.get("timestamp", "")
            if not ts_str:
                continue

            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, AttributeError):
                continue

            if ts < cutoff:
                continue

            # Determine bucket key based on granularity
            if granularity == "hour":
                key = ts.strftime("%Y-%m-%d %H:00")
            elif granularity == "week":
                # Start of week (Monday)
                week_start = ts - timedelta(days=ts.weekday())
                key = week_start.strftime("%Y-%m-%d")
            elif granularity == "month":
                key = ts.strftime("%Y-%m")
            else:  # day
                key = ts.strftime("%Y-%m-%d")

            grouped_data[key]["emissions_kg"] += record.get("emissions_kg_co2", 0)
            grouped_data[key]["energy_kwh"] += record.get("energy_consumed_kwh", 0)
            grouped_data[key]["job_count"] += 1

            job_type = record.get("job_type", "training")
            if job_type == "inference":
                grouped_data[key]["inference"] += 1
            else:
                grouped_data[key]["training"] += 1

        # Sort and format data points
        sorted_keys = sorted(grouped_data.keys())
        data_points = [
            {
                "date": key,
                "emissions_kg": round(grouped_data[key]["emissions_kg"], 6),
                "energy_kwh": round(grouped_data[key]["energy_kwh"], 6),
                "job_count": grouped_data[key]["job_count"],
                "training_jobs": grouped_data[key]["training"],
                "inference_jobs": grouped_data[key]["inference"],
            }
            for key in sorted_keys
        ]

        # Calculate totals for the period
        totals = {
            "emissions_kg": sum(d["emissions_kg"] for d in data_points),
            "energy_kwh": sum(d["energy_kwh"] for d in data_points),
            "job_count": sum(d["job_count"] for d in data_points),
        }

        return {
            "period": period,
            "granularity": granularity,
            "data_points": data_points,
            "totals": totals,
        }

    except Exception as e:
        import logging

        logging.warning(f"Could not load emissions trends: {e}")
        return {
            "period": period,
            "granularity": granularity,
            "data_points": [],
            "totals": {"emissions_kg": 0, "energy_kwh": 0, "job_count": 0},
            "error": str(e),
        }


@router.get("/analytics/comparisons")
async def get_emissions_comparisons():
    """
    Get emissions comparisons by model, job type, and efficiency metrics.

    Returns data for comparing different models and configurations.
    """
    try:
        from model_garden.carbon import get_emissions_db

        storage = get_storage_manager()
        training_jobs = storage.load_training_jobs()

        db = get_emissions_db()
        all_emissions = db.get_all_emissions()

        if not all_emissions:
            return {
                "by_model": [],
                "by_type": {"training": {}, "inference": {}},
                "efficiency_ranking": [],
                "top_emitters": [],
            }

        # Group by model
        by_model: dict[str, dict] = defaultdict(
            lambda: {
                "emissions_kg": 0.0,
                "energy_kwh": 0.0,
                "duration_seconds": 0.0,
                "job_count": 0,
            }
        )

        # Group by type
        by_type: dict[str, dict] = defaultdict(
            lambda: {
                "emissions_kg": 0.0,
                "energy_kwh": 0.0,
                "duration_seconds": 0.0,
                "job_count": 0,
            }
        )

        # Individual job efficiency
        job_efficiency = []

        for record in all_emissions:
            job_id = record.get("job_id", "unknown")
            job_type = record.get("job_type", "training")

            # Get model name from record or training jobs
            model_name = record.get("model_name", "")
            if not model_name and job_id in training_jobs:
                model_name = training_jobs[job_id].get("base_model", "Unknown")
            if not model_name:
                model_name = "Unknown"

            # Simplify model name (just take the last part)
            if "/" in model_name:
                model_name = model_name.split("/")[-1]

            emissions = record.get("emissions_kg_co2", 0)
            energy = record.get("energy_consumed_kwh", 0)
            duration = record.get("duration_seconds", 0)

            # Aggregate by model
            by_model[model_name]["emissions_kg"] += emissions
            by_model[model_name]["energy_kwh"] += energy
            by_model[model_name]["duration_seconds"] += duration
            by_model[model_name]["job_count"] += 1

            # Aggregate by type
            by_type[job_type]["emissions_kg"] += emissions
            by_type[job_type]["energy_kwh"] += energy
            by_type[job_type]["duration_seconds"] += duration
            by_type[job_type]["job_count"] += 1

            # Calculate efficiency (emissions per hour)
            if duration > 0:
                emissions_per_hour = (emissions / duration) * 3600
                job_efficiency.append(
                    {
                        "job_id": job_id,
                        "model": model_name,
                        "job_type": job_type,
                        "emissions_kg": round(emissions, 6),
                        "duration_hours": round(duration / 3600, 2),
                        "emissions_per_hour": round(emissions_per_hour, 6),
                        "energy_kwh": round(energy, 4),
                    }
                )

        # Format model comparisons
        model_comparisons = [
            {
                "model": model,
                "emissions_kg": round(data["emissions_kg"], 6),
                "energy_kwh": round(data["energy_kwh"], 4),
                "duration_hours": round(data["duration_seconds"] / 3600, 2),
                "job_count": data["job_count"],
                "avg_emissions_per_job": round(
                    data["emissions_kg"] / data["job_count"] if data["job_count"] > 0 else 0, 6
                ),
            }
            for model, data in sorted(
                by_model.items(), key=lambda x: x[1]["emissions_kg"], reverse=True
            )
        ]

        # Format type comparisons
        type_comparisons = {
            job_type: {
                "emissions_kg": round(data["emissions_kg"], 6),
                "energy_kwh": round(data["energy_kwh"], 4),
                "duration_hours": round(data["duration_seconds"] / 3600, 2),
                "job_count": data["job_count"],
            }
            for job_type, data in by_type.items()
        }

        # Efficiency ranking (lowest emissions per hour is best)
        efficiency_ranking = sorted(job_efficiency, key=lambda x: x["emissions_per_hour"])[
            :10
        ]  # Top 10 most efficient

        # Top emitters
        top_emitters = sorted(job_efficiency, key=lambda x: x["emissions_kg"], reverse=True)[:10]

        return {
            "by_model": model_comparisons,
            "by_type": type_comparisons,
            "efficiency_ranking": efficiency_ranking,
            "top_emitters": top_emitters,
        }

    except Exception as e:
        import logging

        logging.warning(f"Could not load emissions comparisons: {e}")
        return {
            "by_model": [],
            "by_type": {},
            "efficiency_ranking": [],
            "top_emitters": [],
            "error": str(e),
        }


@router.get("/analytics/recommendations")
async def get_carbon_recommendations():
    """
    Get personalized recommendations for reducing carbon emissions.

    Analyzes usage patterns and provides actionable tips.
    """
    try:
        from model_garden.carbon import get_emissions_db

        storage = get_storage_manager()
        training_jobs = storage.load_training_jobs()

        db = get_emissions_db()
        all_emissions = db.get_all_emissions()

        recommendations = []
        insights = []

        if not all_emissions:
            return {
                "recommendations": [
                    {
                        "id": "start-tracking",
                        "priority": "info",
                        "title": "Start Tracking Your Carbon Footprint",
                        "description": "Run training jobs or inference to begin tracking emissions.",
                        "potential_savings": None,
                        "action": "Start a training job to see emissions data",
                    }
                ],
                "insights": [],
                "summary": {
                    "total_potential_savings_kg": 0,
                    "efficiency_score": 100,
                    "recommendation_count": 1,
                },
            }

        # Analyze patterns
        total_emissions = sum(r.get("emissions_kg_co2", 0) for r in all_emissions)
        total_duration = sum(r.get("duration_seconds", 0) for r in all_emissions)
        job_count = len(all_emissions)

        # Insight: Total impact
        insights.append(
            {
                "type": "total_impact",
                "title": "Your Total Carbon Impact",
                "value": f"{total_emissions:.4f} kg CO₂",
                "context": f"From {job_count} jobs over {total_duration / 3600:.1f} hours",
            }
        )

        # Group by model size (heuristic based on name)
        large_model_jobs = []
        small_model_jobs = []

        for record in all_emissions:
            model_name = record.get("model_name", "").lower()
            job_id = record.get("job_id", "")
            if job_id in training_jobs:
                model_name = training_jobs[job_id].get("base_model", "").lower()

            emissions = record.get("emissions_kg_co2", 0)

            # Heuristic: large models usually have 70b, 72b, 32b, etc in name
            if any(size in model_name for size in ["70b", "72b", "32b", "34b", "65b", "13b"]):
                large_model_jobs.append(emissions)
            elif any(size in model_name for size in ["1b", "3b", "7b", "8b"]):
                small_model_jobs.append(emissions)

        # Recommendation: Use smaller models when possible
        if large_model_jobs:
            avg_large = sum(large_model_jobs) / len(large_model_jobs)
            avg_small = (
                sum(small_model_jobs) / len(small_model_jobs)
                if small_model_jobs
                else avg_large * 0.3
            )

            if avg_large > avg_small * 2:
                potential_savings = (avg_large - avg_small) * len(large_model_jobs)
                recommendations.append(
                    {
                        "id": "use-smaller-models",
                        "priority": "high",
                        "title": "Consider Smaller Models for Development",
                        "description": f"Your large model jobs average {avg_large:.4f} kg CO₂. Using smaller models for experimentation could reduce emissions significantly.",
                        "potential_savings_kg": round(potential_savings, 4),
                        "action": "Try 3B-7B parameter models for prototyping before scaling up",
                    }
                )

        # Recommendation: Batch size optimization
        short_jobs = [r for r in all_emissions if r.get("duration_seconds", 0) < 300]  # < 5 min
        if len(short_jobs) > 5:
            recommendations.append(
                {
                    "id": "consolidate-short-jobs",
                    "priority": "medium",
                    "title": "Consolidate Short Training Jobs",
                    "description": f"You have {len(short_jobs)} jobs under 5 minutes. Consider batching experiments or using larger datasets to reduce startup overhead.",
                    "potential_savings_kg": round(len(short_jobs) * 0.001, 4),  # Rough estimate
                    "action": "Group similar experiments into single longer runs",
                }
            )

        # Recommendation: Inference optimization
        inference_jobs = [r for r in all_emissions if r.get("job_type") == "inference"]
        if inference_jobs:
            total_inference_emissions = sum(r.get("emissions_kg_co2", 0) for r in inference_jobs)
            insights.append(
                {
                    "type": "inference_impact",
                    "title": "Inference Carbon Footprint",
                    "value": f"{total_inference_emissions:.6f} kg CO₂",
                    "context": f"From {len(inference_jobs)} inference sessions",
                }
            )

            if total_inference_emissions > total_emissions * 0.3:
                recommendations.append(
                    {
                        "id": "optimize-inference",
                        "priority": "medium",
                        "title": "Optimize Inference Efficiency",
                        "description": "Inference accounts for a significant portion of your emissions. Consider using quantization (AWQ, GPTQ) or smaller models for production.",
                        "potential_savings_kg": round(total_inference_emissions * 0.3, 4),
                        "action": "Enable 4-bit or 8-bit quantization for inference",
                    }
                )

        # Recommendation: Time-of-day optimization
        # Note: This would require tracking time of day in emissions data
        recommendations.append(
            {
                "id": "green-energy-timing",
                "priority": "low",
                "title": "Schedule Jobs During Low-Carbon Hours",
                "description": "Running compute-intensive jobs during off-peak hours when renewable energy is more available can reduce carbon intensity.",
                "potential_savings_kg": round(total_emissions * 0.1, 4),
                "action": "Schedule large training jobs for early morning or late evening",
            }
        )

        # Recommendation: Use LoRA for fine-tuning
        full_finetune_jobs = []
        lora_jobs = []

        for record in all_emissions:
            job_id = record.get("job_id", "")
            if job_id in training_jobs:
                job = training_jobs[job_id]
                if job.get("lora_config"):
                    lora_jobs.append(record.get("emissions_kg_co2", 0))
                else:
                    full_finetune_jobs.append(record.get("emissions_kg_co2", 0))

        if full_finetune_jobs and not lora_jobs:
            recommendations.append(
                {
                    "id": "use-lora",
                    "priority": "high",
                    "title": "Use LoRA for Efficient Fine-tuning",
                    "description": "LoRA (Low-Rank Adaptation) can reduce training time and memory by 60-80% while maintaining quality. You haven't used LoRA yet.",
                    "potential_savings_kg": round(sum(full_finetune_jobs) * 0.6, 4),
                    "action": "Enable LoRA in your next training job",
                }
            )

        # Calculate efficiency score (0-100)
        # Based on: using small models, short jobs efficiency, LoRA usage
        efficiency_score = 70  # Base score
        if small_model_jobs and len(small_model_jobs) > len(large_model_jobs):
            efficiency_score += 10
        if lora_jobs:
            efficiency_score += 10
        if len(short_jobs) < job_count * 0.3:
            efficiency_score += 10

        efficiency_score = min(100, efficiency_score)

        # Sort recommendations by priority
        priority_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))

        total_potential_savings = sum(
            r.get("potential_savings_kg", 0)
            for r in recommendations
            if r.get("potential_savings_kg")
        )

        return {
            "recommendations": recommendations,
            "insights": insights,
            "summary": {
                "total_potential_savings_kg": round(total_potential_savings, 4),
                "efficiency_score": efficiency_score,
                "recommendation_count": len(recommendations),
            },
        }

    except Exception as e:
        import logging

        logging.warning(f"Could not generate recommendations: {e}")
        return {
            "recommendations": [],
            "insights": [],
            "summary": {
                "total_potential_savings_kg": 0,
                "efficiency_score": 0,
                "recommendation_count": 0,
            },
            "error": str(e),
        }
