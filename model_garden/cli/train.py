"""Training commands for Model Garden CLI.

Contains:
- train: Fine-tune a text language model
- train-vision: Fine-tune a vision-language model

Both commands use TrainingService for actual training logic, ensuring
consistency with the API and eliminating code duplication.
"""

from typing import Literal, cast

import click

from model_garden.services import TrainingRequest, TrainingService
from model_garden.training.config import (
    LoRAConfig,
    TrainingConfig,
    VisionLoRAConfig,
    VisionTrainingConfig,
)
from model_garden.utils.console import console


@click.command()
@click.option(
    "--base-model",
    "-m",
    required=True,
    help="Base model to fine-tune (HuggingFace ID or local path)",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    help="Path to dataset file (JSONL, JSON, CSV) or HuggingFace dataset ID",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    help="Directory to save the fine-tuned model",
)
@click.option(
    "--epochs",
    "-e",
    default=3,
    type=int,
    help="Number of training epochs",
)
@click.option(
    "--batch-size",
    "-b",
    default=2,
    type=int,
    help="Training batch size per device",
)
@click.option(
    "--learning-rate",
    "-lr",
    default=2e-4,
    type=float,
    help="Learning rate",
)
@click.option(
    "--max-seq-length",
    default=2048,
    type=int,
    help="Maximum sequence length",
)
@click.option(
    "--lora-r",
    default=16,
    type=int,
    help="LoRA rank",
)
@click.option(
    "--lora-alpha",
    default=16,
    type=int,
    help="LoRA alpha parameter (scaling factor, typically equal to lora-r)",
)
@click.option(
    "--lora-dropout",
    default=0.0,
    type=float,
    help="LoRA dropout rate (0.0-0.3, higher = more regularization)",
)
@click.option(
    "--lora-bias",
    type=click.Choice(["none", "all", "lora_only"]),
    default="none",
    help="How to handle bias in LoRA layers",
)
@click.option(
    "--gradient-accumulation-steps",
    default=4,
    type=int,
    help="Gradient accumulation steps",
)
@click.option(
    "--max-steps",
    default=-1,
    type=int,
    help="Maximum training steps (-1 for full epochs)",
)
@click.option(
    "--logging-steps",
    default=10,
    type=int,
    help="Log every N steps",
)
@click.option(
    "--save-steps",
    default=100,
    type=int,
    help="Save checkpoint every N steps",
)
@click.option(
    "--save-method",
    type=click.Choice(["lora", "merged_16bit", "merged_4bit"]),
    default="merged_16bit",
    help="How to save the final model",
)
@click.option(
    "--instruction-field",
    default="instruction",
    help="Dataset field name for instructions",
)
@click.option(
    "--input-field",
    default="input",
    help="Dataset field name for inputs",
)
@click.option(
    "--output-field",
    default="output",
    help="Dataset field name for outputs",
)
@click.option(
    "--weight-decay",
    default=0.01,
    type=float,
    help="Weight decay for regularization (0.0-0.1)",
)
@click.option(
    "--lr-scheduler-type",
    type=click.Choice(["linear", "cosine", "constant", "constant_with_warmup", "polynomial"]),
    default="linear",
    help="Learning rate scheduler type",
)
@click.option(
    "--max-grad-norm",
    default=1.0,
    type=float,
    help="Maximum gradient norm for clipping",
)
@click.option(
    "--adam-beta1",
    default=0.9,
    type=float,
    help="Beta1 parameter for Adam optimizer",
)
@click.option(
    "--adam-beta2",
    default=0.999,
    type=float,
    help="Beta2 parameter for Adam optimizer",
)
@click.option(
    "--adam-epsilon",
    default=1e-8,
    type=float,
    help="Epsilon parameter for Adam optimizer",
)
@click.option(
    "--dataloader-num-workers",
    default=0,
    type=int,
    help="Number of dataloader workers (0 = main process only)",
)
@click.option(
    "--eval-strategy",
    type=click.Choice(["no", "steps", "epoch"]),
    default="steps",
    help="Evaluation strategy",
)
@click.option(
    "--save-total-limit",
    default=3,
    type=int,
    help="Maximum number of checkpoints to keep",
)
@click.option(
    "--from-hub",
    is_flag=True,
    help="Load dataset from HuggingFace Hub instead of local file",
)
@click.option(
    "--use-gradient-checkpointing",
    type=click.Choice(["unsloth", "true", "false"]),
    default="unsloth",
    help="Gradient checkpointing mode: 'unsloth' (most memory efficient), 'true' (better quality), 'false' (best quality, most memory)",
)
@click.option(
    "--load-in-16bit",
    is_flag=True,
    help="Load model in 16-bit precision instead of 4-bit (better quality, 4x more memory)",
)
@click.option(
    "--load-in-8bit",
    is_flag=True,
    help="Load model in 8-bit precision instead of 4-bit (balanced quality/memory)",
)
@click.option(
    "--use-rslora",
    is_flag=True,
    help="Use rank-stabilized LoRA (recommended for high ranks >= 32)",
)
@click.option(
    "--optim",
    type=click.Choice(["adamw_8bit", "adamw_torch", "adamw_torch_fused", "adafactor"]),
    default="adamw_8bit",
    help="Optimizer: 'adamw_8bit' (memory efficient), 'adamw_torch' (better quality), 'adamw_torch_fused' (best quality/speed)",
)
@click.option(
    "--quality-mode",
    is_flag=True,
    help="Enable quality-optimized settings (16-bit precision, standard gradient checkpointing, better optimizer)",
)
@click.option(
    "--backend",
    type=str,
    default=None,
    help="Training backend to use (default: auto-select best available). Use 'model-garden list-backends' to see available options.",
)
def train(
    base_model: str,
    dataset: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: str,
    gradient_accumulation_steps: int,
    max_steps: int,
    logging_steps: int,
    save_steps: int,
    save_method: str,
    instruction_field: str,
    input_field: str,
    output_field: str,
    weight_decay: float,
    lr_scheduler_type: str,
    max_grad_norm: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    dataloader_num_workers: int,
    eval_strategy: str,
    save_total_limit: int,
    from_hub: bool,
    use_gradient_checkpointing: str,
    load_in_16bit: bool,
    load_in_8bit: bool,
    use_rslora: bool,
    optim: str,
    quality_mode: bool,
    backend: str | None,
) -> None:
    """Fine-tune a language model.

    Uses the best available backend (Unsloth if installed, otherwise Transformers).

    Example:

        \b
        # Train with default (memory-optimized) settings
        uv run model-garden train \\
            --base-model unsloth/tinyllama-bnb-4bit \\
            --dataset ./data/train.jsonl \\
            --output-dir ./models/my-model \\
            --epochs 3

        \b
        # Train with quality-optimized settings (uses more memory)
        uv run model-garden train \\
            --base-model unsloth/llama-3.1-8b \\
            --dataset ./data/train.jsonl \\
            --output-dir ./models/my-model \\
            --quality-mode \\
            --lora-r 64 \\
            --epochs 3

        \b
        # Train with HuggingFace Hub dataset
        uv run model-garden train \\
            --base-model unsloth/tinyllama-bnb-4bit \\
            --dataset yahma/alpaca-cleaned \\
            --output-dir ./models/my-model \\
            --from-hub
    """
    # Validate precision settings
    if load_in_16bit and load_in_8bit:
        console.print("[red]Error: Cannot use both --load-in-16bit and --load-in-8bit[/red]")
        raise click.Abort()

    # Convert gradient checkpointing string to appropriate value
    if use_gradient_checkpointing == "unsloth":
        gc_value = "unsloth"
    elif use_gradient_checkpointing == "true":
        gc_value = True
    else:  # "false"
        gc_value = False

    # Build training config
    training_config = TrainingConfig(
        output_dir=output_dir,
        num_epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        optim=optim,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        max_grad_norm=max_grad_norm,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        dataloader_num_workers=dataloader_num_workers,
        eval_strategy=eval_strategy,
        save_total_limit=save_total_limit,
    )

    # Build LoRA config
    lora_config = LoRAConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=cast(Literal["none", "all", "lora_only"], lora_bias),
        use_gradient_checkpointing=gc_value,
        use_rslora=use_rslora,
    )

    # Build training request - ALL logic now in TrainingService
    request = TrainingRequest(
        name=f"cli-train-{base_model.split('/')[-1]}",
        base_model=base_model,
        dataset_path=dataset,
        output_dir=output_dir,
        is_vision=False,
        from_hub=from_hub,
        training_config=training_config,
        lora_config=lora_config,
        quality_mode=quality_mode,
        load_in_4bit=not (load_in_16bit or load_in_8bit),
        load_in_8bit=load_in_8bit,
        save_method=cast(Literal["lora", "merged_16bit", "merged_4bit"], save_method),
        backend=backend,
        instruction_field=instruction_field,
        input_field=input_field,
        output_field=output_field,
    )

    # Execute training through unified service
    service = TrainingService()
    result = service.train(request)

    if not result.success:
        console.print(f"\n[bold red]❌ Training failed: {result.error}[/bold red]\n")
        raise click.Abort()


@click.command()
@click.option(
    "--base-model",
    "-m",
    default="Qwen/Qwen2.5-VL-3B-Instruct",
    help="Vision-language model to fine-tune",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    help="Path to dataset file or HuggingFace dataset identifier",
)
@click.option(
    "--from-hub",
    is_flag=True,
    help="Load dataset from HuggingFace Hub instead of local file",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    help="Directory to save the fine-tuned model",
)
@click.option(
    "--epochs",
    "-e",
    default=3,
    type=int,
    help="Number of training epochs",
)
@click.option(
    "--batch-size",
    "-b",
    default=1,
    type=int,
    help="Training batch size per device (use 1 for vision models)",
)
@click.option(
    "--learning-rate",
    "-lr",
    default=2e-5,
    type=float,
    help="Learning rate (lower for vision models)",
)
@click.option(
    "--max-seq-length",
    default=2048,
    type=int,
    help="Maximum sequence length",
)
@click.option(
    "--lora-r",
    default=16,
    type=int,
    help="LoRA rank",
)
@click.option(
    "--lora-alpha",
    default=16,
    type=int,
    help="LoRA alpha parameter (scaling factor, typically equal to lora-r)",
)
@click.option(
    "--lora-dropout",
    default=0.0,
    type=float,
    help="LoRA dropout rate (0.0-0.3, higher = more regularization)",
)
@click.option(
    "--lora-bias",
    type=click.Choice(["none", "all", "lora_only"]),
    default="none",
    help="How to handle bias in LoRA layers",
)
@click.option(
    "--gradient-accumulation-steps",
    default=8,
    type=int,
    help="Gradient accumulation steps (higher for vision models)",
)
@click.option(
    "--max-steps",
    default=-1,
    type=int,
    help="Maximum training steps (-1 for full epochs)",
)
@click.option(
    "--logging-steps",
    default=10,
    type=int,
    help="Log every N steps",
)
@click.option(
    "--save-steps",
    default=100,
    type=int,
    help="Save checkpoint every N steps",
)
@click.option(
    "--save-method",
    type=click.Choice(["lora", "merged_16bit", "merged_4bit"]),
    default="merged_16bit",
    help="How to save the final model (default: merged_16bit)",
)
@click.option(
    "--text-field",
    default="text",
    help="Dataset field name for text/questions",
)
@click.option(
    "--image-field",
    default="image",
    help="Dataset field name for image paths",
)
@click.option(
    "--weight-decay",
    default=0.01,
    type=float,
    help="Weight decay for regularization (0.0-0.1)",
)
@click.option(
    "--lr-scheduler-type",
    type=click.Choice(["linear", "cosine", "constant", "constant_with_warmup", "polynomial"]),
    default="cosine",
    help="Learning rate scheduler type (cosine recommended for vision models)",
)
@click.option(
    "--max-grad-norm",
    default=1.0,
    type=float,
    help="Maximum gradient norm for clipping",
)
@click.option(
    "--adam-beta1",
    default=0.9,
    type=float,
    help="Beta1 parameter for Adam optimizer",
)
@click.option(
    "--adam-beta2",
    default=0.999,
    type=float,
    help="Beta2 parameter for Adam optimizer",
)
@click.option(
    "--adam-epsilon",
    default=1e-8,
    type=float,
    help="Epsilon parameter for Adam optimizer",
)
@click.option(
    "--dataloader-num-workers",
    default=0,
    type=int,
    help="Number of dataloader workers (0 = main process only)",
)
@click.option(
    "--eval-strategy",
    type=click.Choice(["no", "steps", "epoch"]),
    default="steps",
    help="Evaluation strategy",
)
@click.option(
    "--save-total-limit",
    default=3,
    type=int,
    help="Maximum number of checkpoints to keep",
)
@click.option(
    "--selective-loss/--no-selective-loss",
    default=False,
    help="Enable selective loss masking for structured outputs (masks JSON structure)",
)
@click.option(
    "--selective-loss-level",
    type=click.Choice(["conservative", "moderate", "aggressive"]),
    default="conservative",
    help="Selective loss masking level (conservative=structure only, moderate=+null, aggressive=+schema keys)",
)
@click.option(
    "--selective-loss-schema-keys",
    default=None,
    help="Comma-separated schema keys to mask (for aggressive mode, e.g., 'Marque,Modele,contents')",
)
@click.option(
    "--selective-loss-masking-strategy",
    type=click.Choice(["epoch_based", "alternating", "weighted"]),
    default="epoch_based",
    help="Masking strategy: 'epoch_based' (enable after epoch threshold), 'alternating' (cycle ON/OFF), or 'weighted' (soft per-token weights)",
)
@click.option(
    "--selective-loss-masking-start-epoch",
    type=float,
    default=0.0,
    help="[epoch_based only] Delay masking until this epoch (0.0=immediate, 0.5=halfway through first epoch).",
)
@click.option(
    "--selective-loss-mask-every-n-steps",
    type=int,
    default=100,
    help="[alternating only] Full cycle length in training steps (default: 100)",
)
@click.option(
    "--selective-loss-mask-for-n-steps",
    type=int,
    default=50,
    help="[alternating only] Steps with masking ON per cycle (default: 50, i.e., 50%% of cycle)",
)
@click.option(
    "--selective-loss-structural-weight",
    type=float,
    default=0.1,
    help="[weighted only] Weight for structural tokens (0.0-1.0, default: 0.1). Lower = less emphasis on structure.",
)
@click.option(
    "--selective-loss-verbose/--no-selective-loss-verbose",
    default=False,
    help="Print selective loss masking statistics during training",
)
@click.option(
    "--quality-mode",
    is_flag=True,
    help="🏆 Enable quality-optimized settings (16-bit precision, better optimizer, standard gradient checkpointing, RSLoRA for high ranks). Uses ~4x more VRAM than default.",
)
@click.option(
    "--load-in-16bit",
    is_flag=True,
    help="Load model in 16-bit precision (full quality, 4x more memory than 4-bit). Mutually exclusive with --load-in-8bit.",
)
@click.option(
    "--load-in-8bit",
    is_flag=True,
    help="Load model in 8-bit precision (balanced quality/memory, 2x more memory than 4-bit). Mutually exclusive with --load-in-16bit.",
)
@click.option(
    "--use-rslora",
    is_flag=True,
    help="Use rank-stabilized LoRA (recommended for LoRA rank >= 32)",
)
@click.option(
    "--use-gradient-checkpointing",
    type=click.Choice(["unsloth", "true", "false"]),
    default="unsloth",
    help="Gradient checkpointing mode: 'unsloth' (most memory efficient), 'true' (standard, better quality), 'false' (best quality, most memory)",
)
@click.option(
    "--optim",
    type=click.Choice(["adamw_8bit", "adamw_torch", "adamw_torch_fused", "adafactor"]),
    default="adamw_8bit",
    help="Optimizer: adamw_8bit (memory efficient), adamw_torch (better quality), adamw_torch_fused (best quality/speed), adafactor (memory efficient alternative)",
)
@click.option(
    "--finetune-vision-layers/--no-finetune-vision-layers",
    default=True,
    help="Fine-tune vision encoder layers (disable to freeze vision layers and only train language layers)",
)
@click.option(
    "--finetune-language-layers/--no-finetune-language-layers",
    default=True,
    help="Fine-tune language model layers (disable to freeze language layers and only train vision layers)",
)
@click.option(
    "--finetune-attention-modules/--no-finetune-attention-modules",
    default=True,
    help="Fine-tune attention layers (disable for faster training with slightly lower quality)",
)
@click.option(
    "--finetune-mlp-modules/--no-finetune-mlp-modules",
    default=True,
    help="Fine-tune MLP layers (disable for faster training with slightly lower quality)",
)
@click.option(
    "--backend",
    type=str,
    default=None,
    help="Training backend to use (default: auto-select best available). Use 'model-garden list-backends' to see available options.",
)
def train_vision(
    base_model: str,
    dataset: str,
    from_hub: bool,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: str,
    gradient_accumulation_steps: int,
    max_steps: int,
    logging_steps: int,
    save_steps: int,
    save_method: str,
    text_field: str,
    image_field: str,
    weight_decay: float,
    lr_scheduler_type: str,
    max_grad_norm: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    dataloader_num_workers: int,
    eval_strategy: str,
    save_total_limit: int,
    selective_loss: bool,
    selective_loss_level: str,
    selective_loss_schema_keys: str | None,
    selective_loss_masking_strategy: str,
    selective_loss_masking_start_epoch: float,
    selective_loss_mask_every_n_steps: int,
    selective_loss_mask_for_n_steps: int,
    selective_loss_structural_weight: float,
    selective_loss_verbose: bool,
    quality_mode: bool,
    load_in_16bit: bool,
    load_in_8bit: bool,
    use_rslora: bool,
    use_gradient_checkpointing: str,
    optim: str,
    finetune_vision_layers: bool,
    finetune_language_layers: bool,
    finetune_attention_modules: bool,
    finetune_mlp_modules: bool,
    backend: str | None,
) -> None:
    """Fine-tune a vision-language model (e.g., Qwen2.5-VL).

    Uses the best available backend (Unsloth if installed, otherwise Transformers).

    Examples:

        \b
        # Train with local dataset
        uv run model-garden train-vision \\
            --base-model Qwen/Qwen2.5-VL-3B-Instruct \\
            --dataset ./data/vision_dataset.jsonl \\
            --output-dir ./models/my-vision-model \\
            --epochs 3 \\
            --batch-size 1

        \b
        # Train with HuggingFace Hub dataset
        uv run model-garden train-vision \\
            --base-model Qwen/Qwen2.5-VL-3B-Instruct \\
            --dataset Barth371/train_pop_valet_no_wrong_doc \\
            --from-hub \\
            --output-dir ./models/form-extraction-model \\
            --max-steps 100

    Dataset formats:

        Local JSONL:
            {"text": "What is in this image?", "image": "/path/to/img.jpg", "response": "A cat"}

        HuggingFace Hub (OpenAI messages format):
            {"messages": [{"role": "user", "content": [{"type": "image", "image": "data:image/jpeg;base64,..."}]}]}
    """
    # Validate precision settings
    if load_in_16bit and load_in_8bit:
        console.print("[red]Error: Cannot use both --load-in-16bit and --load-in-8bit[/red]")
        raise click.Abort()

    # Convert gradient checkpointing string to appropriate value
    if use_gradient_checkpointing == "unsloth":
        gc_value = "unsloth"
    elif use_gradient_checkpointing == "true":
        gc_value = True
    else:  # "false"
        gc_value = False

    # Parse schema keys if provided
    schema_keys_list = None
    if selective_loss_schema_keys:
        schema_keys_list = [k.strip() for k in selective_loss_schema_keys.split(",")]

    # Build vision training config
    training_config = VisionTrainingConfig(
        output_dir=output_dir,
        num_epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        optim=optim,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        max_grad_norm=max_grad_norm,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        dataloader_num_workers=dataloader_num_workers,
        eval_strategy=eval_strategy,
        save_total_limit=save_total_limit,
        selective_loss=selective_loss,
        selective_loss_level=selective_loss_level,
        selective_loss_schema_keys=schema_keys_list,
        selective_loss_masking_strategy=selective_loss_masking_strategy,
        selective_loss_masking_start_epoch=selective_loss_masking_start_epoch,
        selective_loss_mask_every_n_steps=selective_loss_mask_every_n_steps,
        selective_loss_mask_for_n_steps=selective_loss_mask_for_n_steps,
        selective_loss_structural_weight=selective_loss_structural_weight,
        selective_loss_verbose=selective_loss_verbose,
    )

    # Build vision LoRA config
    lora_config = VisionLoRAConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=cast(Literal["none", "all", "lora_only"], lora_bias),
        use_gradient_checkpointing=gc_value,
        use_rslora=use_rslora,
        finetune_vision_layers=finetune_vision_layers,
        finetune_language_layers=finetune_language_layers,
        finetune_attention_modules=finetune_attention_modules,
        finetune_mlp_modules=finetune_mlp_modules,
    )

    # Build training request - ALL logic now in TrainingService
    request = TrainingRequest(
        name=f"cli-vision-{base_model.split('/')[-1]}",
        base_model=base_model,
        dataset_path=dataset,
        output_dir=output_dir,
        is_vision=True,
        from_hub=from_hub,
        training_config=training_config,
        lora_config=lora_config,
        quality_mode=quality_mode,
        load_in_4bit=not (load_in_16bit or load_in_8bit),
        load_in_8bit=load_in_8bit,
        save_method=cast(Literal["lora", "merged_16bit", "merged_4bit"], save_method),
        backend=backend,
        text_field=text_field,
        image_field=image_field,
    )

    # Execute training through unified service
    service = TrainingService()
    result = service.train(request)

    if not result.success:
        console.print(f"\n[bold red]❌ Training failed: {result.error}[/bold red]\n")
        raise click.Abort()
