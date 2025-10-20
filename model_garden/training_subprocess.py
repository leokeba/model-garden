"""Subprocess-based training execution for memory isolation.

This module runs training jobs in separate subprocesses to ensure complete
memory cleanup between jobs and prevent memory accumulation in the main API process.
"""

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, Optional


def run_training_in_subprocess(job_config: Dict) -> Dict:
    """Execute training in a subprocess with complete memory isolation.
    
    Args:
        job_config: Training job configuration dictionary
        
    Returns:
        Result dictionary with success status and optional error message
    """
    # Import here to avoid importing at module level (which loads unsloth)
    from model_garden.training import ModelTrainer
    from model_garden.vision_training import VisionLanguageTrainer
    from model_garden.memory_management import cleanup_training_resources
    
    try:
        job_id = job_config["id"]
        is_vision = job_config.get("is_vision", False)
        from_hub = job_config.get("from_hub", False)
        validation_from_hub = job_config.get("validation_from_hub", False)
        validation_dataset_path = job_config.get("validation_dataset_path")
        
        # Handle quality mode settings
        quality_mode = job_config.get("quality_mode", False)
        load_in_16bit = job_config.get("load_in_16bit", False)
        load_in_8bit = job_config.get("load_in_8bit", False)
        
        # Apply quality mode overrides
        if quality_mode:
            print("🎯 Quality mode enabled - using higher precision settings")
            load_in_16bit = True
            load_in_8bit = False
        
        # Determine quantization
        load_in_4bit = not (load_in_16bit or load_in_8bit)
        
        if is_vision:
            # Vision-language model training
            from model_garden.vision_training import VisionLanguageTrainer
            
            trainer = VisionLanguageTrainer(
                base_model=job_config["base_model"],
                max_seq_length=job_config["hyperparameters"].get("max_seq_length", 2048),
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )
            
            trainer.load_model()
            
            lora_config = job_config["lora_config"]
            trainer.prepare_for_training(
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("lora_alpha", 16),
                lora_dropout=lora_config.get("lora_dropout", 0.0),
                lora_bias=lora_config.get("lora_bias", "none"),
                use_rslora=lora_config.get("use_rslora", False),
                use_gradient_checkpointing=lora_config.get("use_gradient_checkpointing", "unsloth"),
                random_state=lora_config.get("random_state", 42),
                loftq_config=lora_config.get("loftq_config"),
            )
            
            # Load datasets
            train_dataset = trainer.load_dataset(
                dataset_path=job_config["dataset_path"],
                from_hub=from_hub,
                split="train",
            )
            formatted_train_dataset = trainer.format_dataset(train_dataset)
            
            formatted_val_dataset = None
            if validation_dataset_path:
                val_dataset = trainer.load_dataset(
                    dataset_path=validation_dataset_path,
                    from_hub=validation_from_hub,
                    split="validation",
                )
                formatted_val_dataset = trainer.format_dataset(val_dataset)
            
            # Train (no callbacks in subprocess - progress updates handled by parent)
            hyperparams = job_config["hyperparameters"]
            trainer.train(
                dataset=formatted_train_dataset,
                eval_dataset=formatted_val_dataset,
                eval_steps=hyperparams.get("eval_steps"),
                output_dir=job_config["output_dir"],
                job_id=job_id,
                enable_carbon_tracking=True,
                num_train_epochs=hyperparams.get("num_epochs", 3),
                per_device_train_batch_size=hyperparams.get("batch_size", 1),
                gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 8),
                learning_rate=hyperparams.get("learning_rate", 2e-5),
                warmup_steps=hyperparams.get("warmup_steps", 10),
                max_steps=hyperparams.get("max_steps", -1),
                logging_steps=hyperparams.get("logging_steps", 10),
                save_steps=hyperparams.get("save_steps", 100),
                optim=hyperparams.get("optim", "adamw_8bit"),
                weight_decay=hyperparams.get("weight_decay", 0.01),
                lr_scheduler_type=hyperparams.get("lr_scheduler_type", "cosine"),
                max_grad_norm=hyperparams.get("max_grad_norm", 1.0),
                adam_beta1=hyperparams.get("adam_beta1", 0.9),
                adam_beta2=hyperparams.get("adam_beta2", 0.999),
                adam_epsilon=hyperparams.get("adam_epsilon", 1e-8),
                dataloader_num_workers=hyperparams.get("dataloader_num_workers", 0),
                eval_strategy=hyperparams.get("eval_strategy", "steps"),
                load_best_model_at_end=hyperparams.get("load_best_model_at_end", True),
                metric_for_best_model=hyperparams.get("metric_for_best_model", "eval_loss"),
                save_total_limit=hyperparams.get("save_total_limit", 3),
                callbacks=None,  # No callbacks in subprocess
                selective_loss=job_config.get("selective_loss", False),
                selective_loss_level=job_config.get("selective_loss_level", "conservative"),
                selective_loss_schema_keys=job_config.get("selective_loss_schema_keys"),
                selective_loss_masking_start_step=job_config.get("selective_loss_masking_start_step", 0),
                selective_loss_verbose=job_config.get("selective_loss_verbose", False),
            )
            
            # Save model
            save_method = job_config.get("save_method", "merged_16bit")
            trainer.save_model(job_config["output_dir"], save_method=save_method)
            
            # Cleanup
            cleanup_training_resources(
                trainer.model,
                trainer.tokenizer,
                trainer.processor,
                trainer,
                formatted_train_dataset,
                formatted_val_dataset,
                train_dataset,
            )
            
        else:
            # Text-only model training
            from model_garden.training import ModelTrainer
            
            trainer = ModelTrainer(
                base_model=job_config["base_model"],
                max_seq_length=job_config["hyperparameters"].get("max_seq_length", 2048),
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )
            
            trainer.load_model()
            
            lora_config = job_config["lora_config"]
            trainer.prepare_for_training(
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("lora_alpha", 16),
                lora_dropout=lora_config.get("lora_dropout", 0.0),
                lora_bias=lora_config.get("lora_bias", "none"),
                use_rslora=lora_config.get("use_rslora", False),
                use_gradient_checkpointing=lora_config.get("use_gradient_checkpointing", "unsloth"),
                random_state=lora_config.get("random_state", 42),
                loftq_config=lora_config.get("loftq_config"),
            )
            
            # Load datasets
            if from_hub:
                train_dataset = trainer.load_dataset_from_hub(job_config["dataset_path"], split="train")
            else:
                train_dataset = trainer.load_dataset_from_file(job_config["dataset_path"])
            
            train_dataset = trainer.format_dataset(
                train_dataset,
                instruction_field=job_config["hyperparameters"].get("instruction_field", "instruction"),
                input_field=job_config["hyperparameters"].get("input_field", "input"),
                output_field=job_config["hyperparameters"].get("output_field", "output"),
            )
            
            val_dataset = None
            if validation_dataset_path:
                if validation_from_hub:
                    val_dataset = trainer.load_dataset_from_hub(validation_dataset_path, split="validation")
                else:
                    val_dataset = trainer.load_dataset_from_file(validation_dataset_path)
                
                val_dataset = trainer.format_dataset(
                    val_dataset,
                    instruction_field=job_config["hyperparameters"].get("instruction_field", "instruction"),
                    input_field=job_config["hyperparameters"].get("input_field", "input"),
                    output_field=job_config["hyperparameters"].get("output_field", "output"),
                )
            
            # Train
            hyperparams = job_config["hyperparameters"]
            trainer.train(
                dataset=train_dataset,
                eval_dataset=val_dataset,
                eval_steps=hyperparams.get("eval_steps"),
                output_dir=job_config["output_dir"],
                job_id=job_id,
                enable_carbon_tracking=True,
                num_train_epochs=hyperparams.get("num_epochs", 3),
                per_device_train_batch_size=hyperparams.get("batch_size", 2),
                gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 4),
                learning_rate=hyperparams.get("learning_rate", 2e-4),
                warmup_steps=hyperparams.get("warmup_steps", 10),
                max_steps=hyperparams.get("max_steps", -1),
                logging_steps=hyperparams.get("logging_steps", 10),
                save_steps=hyperparams.get("save_steps", 100),
                optim=hyperparams.get("optim", "adamw_8bit"),
                weight_decay=hyperparams.get("weight_decay", 0.01),
                lr_scheduler_type=hyperparams.get("lr_scheduler_type", "linear"),
                max_grad_norm=hyperparams.get("max_grad_norm", 1.0),
                adam_beta1=hyperparams.get("adam_beta1", 0.9),
                adam_beta2=hyperparams.get("adam_beta2", 0.999),
                adam_epsilon=hyperparams.get("adam_epsilon", 1e-8),
                dataloader_num_workers=hyperparams.get("dataloader_num_workers", 0),
                eval_strategy=hyperparams.get("eval_strategy", "steps"),
                load_best_model_at_end=hyperparams.get("load_best_model_at_end", True),
                metric_for_best_model=hyperparams.get("metric_for_best_model", "eval_loss"),
                save_total_limit=hyperparams.get("save_total_limit", 3),
                callbacks=None,
            )
            
            # Save final model
            save_method = hyperparams.get("save_method", "merged_16bit")
            if save_method != "lora":
                trainer.save_model(job_config["output_dir"], save_method=save_method)
            
            # Cleanup
            cleanup_training_resources(
                trainer.model,
                trainer.tokenizer,
                trainer,
                train_dataset,
                val_dataset,
            )
        
        return {"success": True}
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return {"success": False, "error": error_msg}


def _subprocess_worker(job_config: Dict, result_queue: mp.Queue):
    """Worker function that runs in subprocess.
    
    Args:
        job_config: Training job configuration
        result_queue: Queue for returning results to parent process
    """
    try:
        result = run_training_in_subprocess(job_config)
        result_queue.put(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        result_queue.put({"success": False, "error": str(e)})


def execute_training_job_in_subprocess(job_config: Dict, timeout: Optional[int] = None) -> Dict:
    """Execute a training job in a completely isolated subprocess.
    
    This function spawns a new process, runs the training, and returns the result.
    All memory is freed when the subprocess exits.
    
    Args:
        job_config: Training job configuration dictionary
        timeout: Optional timeout in seconds (None = no timeout)
        
    Returns:
        Result dictionary with success status and optional error message
        
    Raises:
        TimeoutError: If training exceeds timeout
        RuntimeError: If subprocess fails to start
    """
    # Use spawn method to ensure clean process state
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    
    # Start subprocess
    process = ctx.Process(
        target=_subprocess_worker,
        args=(job_config, result_queue)
    )
    
    try:
        process.start()
        
        # Wait for completion
        process.join(timeout=timeout)
        
        if process.is_alive():
            # Timeout occurred
            process.terminate()
            process.join(timeout=10)  # Give it 10s to terminate gracefully
            if process.is_alive():
                process.kill()  # Force kill if still alive
            raise TimeoutError(f"Training job timed out after {timeout} seconds")
        
        # Check exit code
        if process.exitcode != 0:
            # Try to get error from queue
            try:
                result = result_queue.get(timeout=1)
                if not result["success"]:
                    return result
            except:
                pass
            
            raise RuntimeError(f"Training subprocess failed with exit code {process.exitcode}")
        
        # Get result from queue
        try:
            result = result_queue.get(timeout=5)
            return result
        except:
            raise RuntimeError("Failed to get result from training subprocess")
        
    finally:
        # Ensure process is terminated
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
