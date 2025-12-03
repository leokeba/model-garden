<script lang="ts">
    import type { TrainingJob } from "$lib/api/client";
    import Card from "$lib/components/Card.svelte";

    interface Props {
        job: TrainingJob;
    }

    let { job }: Props = $props();

    let showAdvancedSettings = $state(false);
</script>

<Card>
    <div class="p-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Configuration</h3>

        <!-- Basic Info -->
        <div class="space-y-3 text-sm mb-6">
            <div>
                <dt class="block text-gray-700 font-medium">Model Type</dt>
                <dd>{job.config?.model_type || job.model_type || "text"}</dd>
            </div>

            <div>
                <dt class="block text-gray-700 font-medium">Training Backend</dt>
                <dd>
                    <span
                        class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium {job.backend === 'unsloth'
                            ? 'bg-green-100 text-green-800'
                            : 'bg-blue-100 text-blue-800'}"
                    >
                        {job.backend || "unsloth"}
                    </span>
                </dd>
            </div>

            <div>
                <dt class="block text-gray-700 font-medium">
                    Validation Dataset
                </dt>
                <dd>
                    {#if job.validation_dataset_path}
                        <span
                            class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800"
                        >
                            ✓ Enabled
                        </span>
                    {:else}
                        <span class="text-gray-500">Not provided</span>
                    {/if}
                </dd>
            </div>

            <div>
                <dt class="block text-gray-700 font-medium">Save Method</dt>
                <dd>{job.save_method || "merged_16bit"}</dd>
            </div>

            <div>
                <dt class="block text-gray-700 font-medium">Quality Mode</dt>
                <dd>
                    {#if job.quality_mode}
                        <span
                            class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800"
                        >
                            ✓ Enabled (16-bit)
                        </span>
                    {:else if job.load_in_16bit}
                        <span
                            class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800"
                        >
                            16-bit Precision
                        </span>
                    {:else if job.load_in_8bit}
                        <span
                            class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800"
                        >
                            8-bit Precision
                        </span>
                    {:else}
                        <span class="text-gray-500">4-bit (Default)</span>
                    {/if}
                </dd>
            </div>

            {#if job.early_stopping_enabled}
                <div>
                    <dt class="block text-gray-700 font-medium">
                        Early Stopping
                    </dt>
                    <dd>
                        <span
                            class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800"
                        >
                            ✓ Enabled (patience: {job.early_stopping_patience ||
                                3})
                        </span>
                    </dd>
                </div>
            {/if}

            {#if job.selective_loss}
                <div>
                    <dt class="block text-gray-700 font-medium">
                        Selective Loss
                    </dt>
                    <dd>
                        <span
                            class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-100 text-indigo-800"
                        >
                            ✓ {job.selective_loss_level || "conservative"}
                        </span>
                    </dd>
                </div>
            {/if}

            {#if job.rerun_from}
                <div>
                    <dt class="block text-gray-700 font-medium">Rerun From</dt>
                    <dd>
                        <a
                            href="/training/{job.rerun_from}"
                            class="text-primary-600 hover:text-primary-800 underline"
                        >
                            {job.rerun_from_name || job.rerun_from}
                        </a>
                    </dd>
                </div>
            {/if}
        </div>

        <!-- Hyperparameters -->
        {#if job.hyperparameters}
            <div class="border-t pt-4 mb-6">
                <h4 class="text-md font-medium text-gray-900 mb-3">
                    Training Hyperparameters
                </h4>
                <div class="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                    <div>
                        <dt class="text-gray-700 font-medium">Learning Rate</dt>
                        <dd class="text-gray-900">
                            {job.hyperparameters.learning_rate}
                        </dd>
                    </div>
                    <div>
                        <dt class="text-gray-700 font-medium">Epochs</dt>
                        <dd class="text-gray-900">
                            {job.hyperparameters.num_epochs}
                        </dd>
                    </div>
                    <div>
                        <dt class="text-gray-700 font-medium">Batch Size</dt>
                        <dd class="text-gray-900">
                            {job.hyperparameters.batch_size}
                        </dd>
                    </div>
                    <div>
                        <dt class="text-gray-700 font-medium">Max Steps</dt>
                        <dd class="text-gray-900">
                            {job.hyperparameters.max_steps || "Auto"}
                        </dd>
                    </div>
                    <div>
                        <dt class="text-gray-700 font-medium">
                            Gradient Accumulation
                        </dt>
                        <dd class="text-gray-900">
                            {job.hyperparameters.gradient_accumulation_steps}
                        </dd>
                    </div>
                    <div>
                        <dt class="text-gray-700 font-medium">Warmup Steps</dt>
                        <dd class="text-gray-900">
                            {job.hyperparameters.warmup_steps}
                        </dd>
                    </div>
                    <div>
                        <dt class="text-gray-700 font-medium">Optimizer</dt>
                        <dd class="text-gray-900">
                            {job.hyperparameters.optim}
                        </dd>
                    </div>
                    <div>
                        <dt class="text-gray-700 font-medium">LR Scheduler</dt>
                        <dd class="text-gray-900">
                            {job.hyperparameters.lr_scheduler_type || "linear"}
                        </dd>
                    </div>
                    {#if job.hyperparameters.weight_decay !== undefined}
                        <div>
                            <dt class="text-gray-700 font-medium">
                                Weight Decay
                            </dt>
                            <dd class="text-gray-900">
                                {job.hyperparameters.weight_decay}
                            </dd>
                        </div>
                    {/if}
                    {#if job.hyperparameters.max_grad_norm !== undefined}
                        <div>
                            <dt class="text-gray-700 font-medium">
                                Max Grad Norm
                            </dt>
                            <dd class="text-gray-900">
                                {job.hyperparameters.max_grad_norm}
                            </dd>
                        </div>
                    {/if}
                    {#if job.hyperparameters.eval_strategy}
                        <div>
                            <dt class="text-gray-700 font-medium">
                                Eval Strategy
                            </dt>
                            <dd class="text-gray-900">
                                {job.hyperparameters.eval_strategy}
                            </dd>
                        </div>
                    {/if}
                    {#if job.hyperparameters.eval_steps}
                        <div>
                            <dt class="text-gray-700 font-medium">
                                Eval Steps
                            </dt>
                            <dd class="text-gray-900">
                                {job.hyperparameters.eval_steps}
                            </dd>
                        </div>
                    {/if}
                </div>
            </div>
        {/if}

        <!-- LoRA Configuration -->
        {#if job.lora_config}
            <div class="border-t pt-4">
                <h4 class="text-md font-medium text-gray-900 mb-3">
                    LoRA Configuration
                </h4>
                <div class="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                    <div>
                        <dt class="text-gray-700 font-medium">Rank (r)</dt>
                        <dd class="text-gray-900">{job.lora_config.r}</dd>
                    </div>
                    <div>
                        <dt class="text-gray-700 font-medium">Alpha</dt>
                        <dd class="text-gray-900">
                            {job.lora_config.lora_alpha}
                        </dd>
                    </div>
                    <div>
                        <dt class="text-gray-700 font-medium">Dropout</dt>
                        <dd class="text-gray-900">
                            {job.lora_config.lora_dropout}
                        </dd>
                    </div>
                    {#if job.lora_config.lora_bias}
                        <div>
                            <dt class="text-gray-700 font-medium">Bias</dt>
                            <dd class="text-gray-900">
                                {job.lora_config.lora_bias}
                            </dd>
                        </div>
                    {/if}
                    {#if job.lora_config.use_rslora !== undefined}
                        <div>
                            <dt class="text-gray-700 font-medium">RSLoRA</dt>
                            <dd class="text-gray-900">
                                {job.lora_config.use_rslora
                                    ? "Enabled"
                                    : "Disabled"}
                            </dd>
                        </div>
                    {/if}
                    {#if job.lora_config.task_type}
                        <div>
                            <dt class="text-gray-700 font-medium">Task Type</dt>
                            <dd class="text-gray-900">
                                {job.lora_config.task_type}
                            </dd>
                        </div>
                    {/if}
                    {#if job.lora_config.target_modules}
                        <div class="col-span-2">
                            <dt class="text-gray-700 font-medium">
                                Target Modules
                            </dt>
                            <dd class="text-gray-900 text-xs">
                                {Array.isArray(job.lora_config.target_modules)
                                    ? job.lora_config.target_modules.join(", ")
                                    : job.lora_config.target_modules}
                            </dd>
                        </div>
                    {/if}
                </div>
            </div>
        {/if}

        <!-- Advanced Settings (Collapsible) -->
        {#if job.hyperparameters && (job.hyperparameters.adam_beta1 !== undefined || job.hyperparameters.dataloader_num_workers !== undefined || job.hyperparameters.metric_for_best_model !== undefined || job.early_stopping_enabled || job.selective_loss)}
            <div class="border-t pt-4">
                <button
                    class="flex items-center justify-between w-full text-left"
                    onclick={() =>
                        (showAdvancedSettings = !showAdvancedSettings)}
                >
                    <h4 class="text-md font-medium text-gray-900">
                        Advanced Settings
                    </h4>
                    <svg
                        class="h-5 w-5 text-gray-400 transform transition-transform duration-200 {showAdvancedSettings
                            ? 'rotate-180'
                            : ''}"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                    >
                        <path
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            stroke-width="2"
                            d="M19 9l-7 7-7-7"
                        />
                    </svg>
                </button>

                {#if showAdvancedSettings}
                    <div class="mt-3 space-y-4">
                        <!-- Optimizer Settings -->
                        {#if job.hyperparameters.adam_beta1 !== undefined || job.hyperparameters.adam_beta2 !== undefined || job.hyperparameters.adam_epsilon !== undefined}
                            <div>
                                <h5
                                    class="text-sm font-medium text-gray-800 mb-2"
                                >
                                    Optimizer Parameters
                                </h5>
                                <div
                                    class="grid grid-cols-2 gap-x-4 gap-y-1 text-sm"
                                >
                                    {#if job.hyperparameters.adam_beta1 !== undefined}
                                        <div>
                                            <dt class="text-gray-600">Beta1</dt>
                                            <dd class="text-gray-900">
                                                {job.hyperparameters.adam_beta1}
                                            </dd>
                                        </div>
                                    {/if}
                                    {#if job.hyperparameters.adam_beta2 !== undefined}
                                        <div>
                                            <dt class="text-gray-600">Beta2</dt>
                                            <dd class="text-gray-900">
                                                {job.hyperparameters.adam_beta2}
                                            </dd>
                                        </div>
                                    {/if}
                                    {#if job.hyperparameters.adam_epsilon !== undefined}
                                        <div>
                                            <dt class="text-gray-600">
                                                Epsilon
                                            </dt>
                                            <dd class="text-gray-900">
                                                {job.hyperparameters
                                                    .adam_epsilon}
                                            </dd>
                                        </div>
                                    {/if}
                                </div>
                            </div>
                        {/if}

                        <!-- Dataloader Settings -->
                        {#if job.hyperparameters.dataloader_num_workers !== undefined || job.hyperparameters.dataloader_pin_memory !== undefined}
                            <div>
                                <h5
                                    class="text-sm font-medium text-gray-800 mb-2"
                                >
                                    Dataloader Settings
                                </h5>
                                <div
                                    class="grid grid-cols-2 gap-x-4 gap-y-1 text-sm"
                                >
                                    {#if job.hyperparameters.dataloader_num_workers !== undefined}
                                        <div>
                                            <dt class="text-gray-600">
                                                Workers
                                            </dt>
                                            <dd class="text-gray-900">
                                                {job.hyperparameters
                                                    .dataloader_num_workers}
                                            </dd>
                                        </div>
                                    {/if}
                                    {#if job.hyperparameters.dataloader_pin_memory !== undefined}
                                        <div>
                                            <dt class="text-gray-600">
                                                Pin Memory
                                            </dt>
                                            <dd class="text-gray-900">
                                                {job.hyperparameters
                                                    .dataloader_pin_memory
                                                    ? "Enabled"
                                                    : "Disabled"}
                                            </dd>
                                        </div>
                                    {/if}
                                </div>
                            </div>
                        {/if}

                        <!-- Evaluation Settings -->
                        {#if job.hyperparameters.metric_for_best_model !== undefined || job.hyperparameters.load_best_model_at_end !== undefined || job.hyperparameters.save_total_limit !== undefined}
                            <div>
                                <h5
                                    class="text-sm font-medium text-gray-800 mb-2"
                                >
                                    Evaluation & Saving
                                </h5>
                                <div
                                    class="grid grid-cols-2 gap-x-4 gap-y-1 text-sm"
                                >
                                    {#if job.hyperparameters.metric_for_best_model !== undefined}
                                        <div>
                                            <dt class="text-gray-600">
                                                Best Model Metric
                                            </dt>
                                            <dd class="text-gray-900">
                                                {job.hyperparameters
                                                    .metric_for_best_model}
                                            </dd>
                                        </div>
                                    {/if}
                                    {#if job.hyperparameters.load_best_model_at_end !== undefined}
                                        <div>
                                            <dt class="text-gray-600">
                                                Load Best Model
                                            </dt>
                                            <dd class="text-gray-900">
                                                {job.hyperparameters
                                                    .load_best_model_at_end
                                                    ? "Yes"
                                                    : "No"}
                                            </dd>
                                        </div>
                                    {/if}
                                    {#if job.hyperparameters.save_total_limit !== undefined}
                                        <div>
                                            <dt class="text-gray-600">
                                                Save Limit
                                            </dt>
                                            <dd class="text-gray-900">
                                                {job.hyperparameters
                                                    .save_total_limit} checkpoints
                                            </dd>
                                        </div>
                                    {/if}
                                </div>
                            </div>
                        {/if}

                        <!-- LoRA Advanced Settings -->
                        {#if job.lora_config && (job.lora_config.use_gradient_checkpointing !== undefined || job.lora_config.random_state !== undefined)}
                            <div>
                                <h5
                                    class="text-sm font-medium text-gray-800 mb-2"
                                >
                                    LoRA Advanced
                                </h5>
                                <div
                                    class="grid grid-cols-2 gap-x-4 gap-y-1 text-sm"
                                >
                                    {#if job.lora_config.use_gradient_checkpointing !== undefined}
                                        <div>
                                            <dt class="text-gray-600">
                                                Gradient Checkpointing
                                            </dt>
                                            <dd class="text-gray-900">
                                                {job.lora_config
                                                    .use_gradient_checkpointing}
                                            </dd>
                                        </div>
                                    {/if}
                                    {#if job.lora_config.random_state !== undefined}
                                        <div>
                                            <dt class="text-gray-600">
                                                Random Seed
                                            </dt>
                                            <dd class="text-gray-900">
                                                {job.lora_config.random_state}
                                            </dd>
                                        </div>
                                    {/if}
                                </div>
                            </div>
                        {/if}

                        <!-- Early Stopping Settings -->
                        {#if job.early_stopping_enabled}
                            <div>
                                <h5
                                    class="text-sm font-medium text-gray-800 mb-2"
                                >
                                    Early Stopping
                                </h5>
                                <div
                                    class="grid grid-cols-2 gap-x-4 gap-y-1 text-sm"
                                >
                                    <div>
                                        <dt class="text-gray-600">Enabled</dt>
                                        <dd class="text-gray-900">Yes</dd>
                                    </div>
                                    <div>
                                        <dt class="text-gray-600">Patience</dt>
                                        <dd class="text-gray-900">
                                            {job.early_stopping_patience || 3} evaluations
                                        </dd>
                                    </div>
                                    {#if job.early_stopping_threshold !== undefined}
                                        <div>
                                            <dt class="text-gray-600">
                                                Threshold
                                            </dt>
                                            <dd class="text-gray-900">
                                                {job.early_stopping_threshold}
                                            </dd>
                                        </div>
                                    {/if}
                                </div>
                            </div>
                        {/if}

                        <!-- Selective Loss Settings -->
                        {#if job.selective_loss}
                            <div>
                                <h5
                                    class="text-sm font-medium text-gray-800 mb-2"
                                >
                                    Selective Loss (Structured Outputs)
                                </h5>
                                <div
                                    class="grid grid-cols-2 gap-x-4 gap-y-1 text-sm"
                                >
                                    <div>
                                        <dt class="text-gray-600">Enabled</dt>
                                        <dd class="text-gray-900">Yes</dd>
                                    </div>
                                    <div>
                                        <dt class="text-gray-600">Level</dt>
                                        <dd class="text-gray-900">
                                            {job.selective_loss_level ||
                                                "conservative"}
                                        </dd>
                                    </div>
                                    {#if job.selective_loss_masking_strategy !== undefined}
                                        <div>
                                            <dt class="text-gray-600">
                                                Masking Strategy
                                            </dt>
                                            <dd class="text-gray-900">
                                                {job.selective_loss_masking_strategy}
                                            </dd>
                                        </div>
                                    {/if}
                                    {#if job.selective_loss_structural_weight !== undefined}
                                        <div>
                                            <dt class="text-gray-600">
                                                Structural Weight
                                            </dt>
                                            <dd class="text-gray-900">
                                                {job.selective_loss_structural_weight}
                                            </dd>
                                        </div>
                                    {/if}
                                    {#if job.selective_loss_masking_start_epoch !== undefined}
                                        <div>
                                            <dt class="text-gray-600">
                                                Masking Start Epoch
                                            </dt>
                                            <dd class="text-gray-900">
                                                {job.selective_loss_masking_start_epoch}
                                            </dd>
                                        </div>
                                    {/if}
                                    {#if job.selective_loss_mask_every_n_steps !== undefined}
                                        <div>
                                            <dt class="text-gray-600">
                                                Mask Every N Steps
                                            </dt>
                                            <dd class="text-gray-900">
                                                {job.selective_loss_mask_every_n_steps}
                                            </dd>
                                        </div>
                                    {/if}
                                    {#if job.selective_loss_mask_for_n_steps !== undefined}
                                        <div>
                                            <dt class="text-gray-600">
                                                Mask For N Steps
                                            </dt>
                                            <dd class="text-gray-900">
                                                {job.selective_loss_mask_for_n_steps}
                                            </dd>
                                        </div>
                                    {/if}
                                    {#if job.selective_loss_verbose !== undefined}
                                        <div>
                                            <dt class="text-gray-600">
                                                Verbose Logging
                                            </dt>
                                            <dd class="text-gray-900">
                                                {job.selective_loss_verbose
                                                    ? "Enabled"
                                                    : "Disabled"}
                                            </dd>
                                        </div>
                                    {/if}
                                    {#if job.selective_loss_schema_keys && job.selective_loss_schema_keys.length > 0}
                                        <div class="col-span-2">
                                            <dt class="text-gray-600">
                                                Schema Keys to Mask
                                            </dt>
                                            <dd class="text-gray-900 text-xs">
                                                {Array.isArray(
                                                    job.selective_loss_schema_keys,
                                                )
                                                    ? job.selective_loss_schema_keys.join(
                                                          ", ",
                                                      )
                                                    : job.selective_loss_schema_keys}
                                            </dd>
                                        </div>
                                    {/if}
                                </div>
                            </div>
                        {/if}
                    </div>
                {/if}
            </div>
        {/if}
    </div>
</Card>
