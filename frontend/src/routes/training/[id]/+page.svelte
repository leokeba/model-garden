<script lang="ts">
  import { page } from "$app/stores";
  import { api, type TrainingJob } from "$lib/api/client";
  import Button from "$lib/components/Button.svelte";
  import Card from "$lib/components/Card.svelte";
  import LossChart from "$lib/components/LossChart.svelte";
  import { onDestroy, onMount } from "svelte";

  const jobId = $derived($page.params.id);

  let job: TrainingJob | null = $state(null);
  let loading = $state(true);
  let error = $state("");
  let ws: WebSocket | null = null;
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  let reconnectAttempts = $state(0);
  let maxReconnectAttempts = 5;
  let isConnected = $state(false);
  let logs: string[] = $state([]);
  let logsContainer = $state<HTMLDivElement | null>(null);
  let trainingMetrics = $state<any[]>([]);
  let validationMetrics = $state<any[]>([]);
  let showAdvancedSettings = $state(false);
  let cancelling = $state(false);

  // Get WebSocket URL dynamically
  function getWebSocketUrl(jobId: string): string {
    if (typeof window === "undefined") return "";

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    return `${protocol}//${host}/ws/training/${jobId}`;
  }

  async function loadJob() {
    if (!jobId) return;

    try {
      loading = true;
      const response = await api.getTrainingJob(jobId);
      job = response;
      error = "";

      // Load existing metrics if available
      if (job.metrics) {
        if (job.metrics.training) {
          trainingMetrics = job.metrics.training;
        }
        if (job.metrics.validation) {
          validationMetrics = job.metrics.validation;
        }
      }
    } catch (err) {
      error =
        err instanceof Error ? err.message : "Failed to load training job";
    } finally {
      loading = false;
    }
  }

  function connectWebSocket() {
    if (!jobId || typeof window === "undefined") return;

    // Don't reconnect if job is completed or failed
    if (
      job &&
      (job.status === "completed" ||
        job.status === "failed" ||
        job.status === "cancelled")
    ) {
      return;
    }

    try {
      const wsUrl = getWebSocketUrl(jobId);
      console.log(`Connecting to WebSocket: ${wsUrl}`);

      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log("WebSocket connected");
        isConnected = true;
        reconnectAttempts = 0;
      };

      ws.onmessage = (event) => {
        try {
          const update = JSON.parse(event.data);
          console.log("WebSocket update:", update);

          // Handle different update types
          if (update.type === "status" && job) {
            job.status = update.status;
            if (update.completed_at) {
              job.completed_at = update.completed_at;
            }
          } else if (update.type === "progress" && job) {
            job.progress = update.progress;
            job.current_step = update.progress?.current_step;
            job.total_steps = update.progress?.total_steps;
            job.current_epoch = update.progress?.epoch;
          } else if (update.type === "training_metrics") {
            // Add new training metric point
            trainingMetrics = [...trainingMetrics, update.metrics];
          } else if (update.type === "validation_metrics") {
            // Add new validation metric point
            validationMetrics = [...validationMetrics, update.metrics];
          } else if (update.type === "log") {
            logs = [
              ...logs,
              `[${new Date().toLocaleTimeString()}] ${update.message}`,
            ];
            // Keep only last 100 log lines
            if (logs.length > 100) {
              logs = logs.slice(-100);
            }
            // Auto-scroll to bottom
            setTimeout(() => scrollLogsToBottom(), 10);
          } else if (update.type === "error" && job) {
            job.error_message = update.message;
            error = update.message;
          }
        } catch (err) {
          console.error("Failed to parse WebSocket message:", err);
        }
      };

      ws.onerror = (event) => {
        console.error("WebSocket error:", event);
        isConnected = false;
      };

      ws.onclose = (event) => {
        console.log("WebSocket closed:", event.code, event.reason);
        isConnected = false;
        ws = null;

        // Attempt to reconnect if job is still running
        if (
          job &&
          job.status === "running" &&
          reconnectAttempts < maxReconnectAttempts
        ) {
          reconnectAttempts++;
          console.log(
            `Reconnecting... (attempt ${reconnectAttempts}/${maxReconnectAttempts})`,
          );
          reconnectTimeout = setTimeout(
            () => {
              connectWebSocket();
            },
            Math.min(1000 * Math.pow(2, reconnectAttempts), 30000),
          ); // Exponential backoff, max 30s
        }
      };
    } catch (err) {
      console.error("Failed to create WebSocket:", err);
      isConnected = false;
    }
  }

  function disconnectWebSocket() {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }

    if (ws) {
      ws.close();
      ws = null;
    }

    isConnected = false;
  }

  function scrollLogsToBottom() {
    if (logsContainer) {
      logsContainer.scrollTop = logsContainer.scrollHeight;
    }
  }

  function getStatusColor(status: string) {
    switch (status) {
      case "running":
        return "text-blue-600 bg-blue-100";
      case "completed":
        return "text-green-600 bg-green-100";
      case "failed":
        return "text-red-600 bg-red-100";
      case "cancelled":
        return "text-gray-600 bg-gray-100";
      case "queued":
        return "text-yellow-600 bg-yellow-100";
      default:
        return "text-gray-600 bg-gray-100";
    }
  }

  function formatProgress(progress: any) {
    if (typeof progress === "number") {
      return Math.round(progress * 100);
    }
    if (
      progress &&
      typeof progress.current_step === "number" &&
      typeof progress.total_steps === "number"
    ) {
      // Handle 0/0 case (returns NaN)
      if (progress.total_steps === 0) {
        return 0;
      }
      return Math.round((progress.current_step / progress.total_steps) * 100);
    }
    return 0;
  }

  function formatDate(dateString: string) {
    return new Date(dateString).toLocaleString();
  }

  async function cancelJob() {
    if (!jobId || !job) return;
    
    const confirmMessage = `Are you sure you want to cancel the training job "${job.config?.name || job.name}"? This action cannot be undone.`;
    if (!confirm(confirmMessage)) return;

    try {
      cancelling = true;
      const response = await api.cancelTrainingJob(jobId);
      
      if (response.success) {
        // Reload job to get updated status
        await loadJob();
        // Disconnect WebSocket since job is no longer running
        disconnectWebSocket();
      } else {
        error = response.message || "Failed to cancel training job";
      }
    } catch (err) {
      error = err instanceof Error ? err.message : "Failed to cancel training job";
    } finally {
      cancelling = false;
    }
  }

  onMount(async () => {
    await loadJob();

    // Connect WebSocket if job is running or queued
    if (job && (job.status === "running" || job.status === "queued")) {
      connectWebSocket();
    }
  });

  onDestroy(() => {
    disconnectWebSocket();
  });
</script>

<svelte:head>
  <title>Training Job {jobId} - Model Garden</title>
</svelte:head>

<div class="min-h-screen bg-gray-50">
  <!-- Header -->
  <div class="bg-white shadow">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between items-center py-6">
        <div class="flex items-center">
          <Button href="/training" variant="ghost" size="sm"
            >← Training Jobs</Button
          >
          <h1 class="text-3xl font-bold text-gray-900 ml-4">
            Training Job Details
          </h1>
        </div>
        {#if job}
          <div class="flex items-center gap-3">
            {#if job.status === "running" || job.status === "queued"}
              <Button 
                variant="danger" 
                size="sm"
                onclick={cancelJob}
                loading={cancelling}
                disabled={cancelling}
              >
                {cancelling ? "Cancelling..." : "Cancel"}
              </Button>
            {/if}
            
            {#if job.status === "running"}
              {#if isConnected}
                <span
                  class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-green-600 bg-green-100"
                >
                  <span
                    class="w-2 h-2 bg-green-600 rounded-full mr-1.5 animate-pulse"
                  ></span>
                  Live Updates
                </span>
              {:else}
                <span
                  class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-yellow-600 bg-yellow-100"
                >
                  <span class="w-2 h-2 bg-yellow-600 rounded-full mr-1.5"
                  ></span>
                  Reconnecting...
                </span>
              {/if}
            {/if}
            <span
              class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium {getStatusColor(
                job.status,
              )}"
            >
              {job.status}
            </span>
          </div>
        {/if}
      </div>
    </div>
  </div>

  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    {#if loading}
      <div class="flex justify-center items-center h-64">
        <div
          class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"
        ></div>
      </div>
    {:else if error}
      <Card>
        <div class="text-center py-8">
          <p class="text-red-600 text-lg">{error}</p>
          <Button onclick={loadJob} variant="primary" class="mt-4"
            >Try Again</Button
          >
        </div>
      </Card>
    {:else if job}
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Main Job Info -->
        <div class="lg:col-span-2 space-y-6">
          <Card>
            <div class="p-6">
              <h2 class="text-xl font-semibold text-gray-900 mb-4">
                Job Information
              </h2>

              <div class="grid grid-cols-2 gap-4">
                <div>
                  <dt class="text-sm font-medium text-gray-700">Job ID</dt>
                  <dd class="mt-1 text-sm text-gray-900">
                    {job.job_id || job.id}
                  </dd>
                </div>

                <div>
                  <dt class="text-sm font-medium text-gray-700">Status</dt>
                  <dd class="mt-1">
                    <span
                      class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium {getStatusColor(
                        job.status,
                      )}"
                    >
                      {job.status}
                    </span>
                  </dd>
                </div>

                <div>
                  <dt class="text-sm font-medium text-gray-700">Model Name</dt>
                  <dd class="mt-1 text-sm text-gray-900">
                    {job.config?.name || job.name}
                  </dd>
                </div>

                <div>
                  <dt class="text-sm font-medium text-gray-700">Base Model</dt>
                  <dd class="mt-1 text-sm text-gray-900">
                    {job.config?.base_model || job.base_model}
                  </dd>
                </div>

                <div>
                  <dt class="text-sm font-medium text-gray-700">Dataset</dt>
                  <dd class="mt-1 text-sm text-gray-900">
                    {job.config?.dataset_path || job.dataset_path}
                  </dd>
                </div>

                <div>
                  <dt class="text-sm font-medium text-gray-700">
                    Output Directory
                  </dt>
                  <dd class="mt-1 text-sm text-gray-900">
                    {job.config?.output_dir || job.output_dir}
                  </dd>
                </div>

                <div>
                  <dt class="text-sm font-medium text-gray-700">Created</dt>
                  <dd class="mt-1 text-sm text-gray-900">
                    {formatDate(job.created_at)}
                  </dd>
                </div>

                {#if job.completed_at}
                  <div>
                    <dt class="text-sm font-medium text-gray-700">Completed</dt>
                    <dd class="mt-1 text-sm text-gray-900">
                      {formatDate(job.completed_at)}
                    </dd>
                  </div>
                {/if}
              </div>
            </div>
          </Card>

          <!-- Loss Curves -->
          {#if trainingMetrics.length > 0 || validationMetrics.length > 0}
            <Card>
              <div class="p-6">
                <LossChart
                  {trainingMetrics}
                  {validationMetrics}
                  title="Training & Validation Loss"
                  height={350}
                />

                <!-- Metrics Table -->
                <div class="mt-6">
                  <h4 class="text-sm font-semibold text-gray-900 mb-3">
                    Recent Metrics
                  </h4>
                  <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                      <thead class="bg-gray-50">
                        <tr>
                          <th
                            scope="col"
                            class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                          >
                            Step
                          </th>
                          <th
                            scope="col"
                            class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                          >
                            Training Loss
                          </th>
                          {#if validationMetrics.length > 0}
                            <th
                              scope="col"
                              class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                            >
                              Validation Loss
                            </th>
                          {/if}
                          <th
                            scope="col"
                            class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                          >
                            Learning Rate
                          </th>
                        </tr>
                      </thead>
                      <tbody class="bg-white divide-y divide-gray-200">
                        {#each trainingMetrics.slice(-10).reverse() as metric}
                          <tr>
                            <td
                              class="px-3 py-2 whitespace-nowrap text-sm text-gray-900"
                            >
                              {metric.step}
                            </td>
                            <td
                              class="px-3 py-2 whitespace-nowrap text-sm text-gray-900"
                            >
                              {metric.loss.toFixed(4)}
                            </td>
                            {#if validationMetrics.length > 0}
                              <td
                                class="px-3 py-2 whitespace-nowrap text-sm text-gray-900"
                              >
                                {#if validationMetrics.find((v) => v.step === metric.step)}
                                  {validationMetrics
                                    .find((v) => v.step === metric.step)
                                    ?.loss.toFixed(4)}
                                {:else}
                                  -
                                {/if}
                              </td>
                            {/if}
                            <td
                              class="px-3 py-2 whitespace-nowrap text-sm text-gray-900"
                            >
                              {metric.learning_rate
                                ? metric.learning_rate.toExponential(2)
                                : "-"}
                            </td>
                          </tr>
                        {/each}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </Card>
          {/if}

          <!-- Progress -->
          {#if job.status === "running" && job.progress}
            <Card>
              <div class="p-6">
                <h2 class="text-xl font-semibold text-gray-900 mb-4">
                  Training Progress
                </h2>

                <div class="space-y-4">
                  <div>
                    <div
                      class="flex justify-between text-sm text-gray-700 mb-1"
                    >
                      <span>Progress</span>
                      <span>{formatProgress(job.progress)}%</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2">
                      <div
                        class="bg-primary-600 h-2 rounded-full transition-all duration-300"
                        style="width: {formatProgress(job.progress)}%"
                      ></div>
                    </div>
                  </div>

                  {#if job.current_step && job.total_steps}
                    <div class="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <dt class="block text-gray-700">Current Step</dt>
                        <dd class="font-semibold">{job.current_step}</dd>
                      </div>
                      <div>
                        <dt class="block text-gray-700">Total Steps</dt>
                        <dd class="font-semibold">{job.total_steps}</dd>
                      </div>
                    </div>
                  {/if}

                  {#if job.current_epoch}
                    <div class="text-sm">
                      <dt class="block text-gray-700">Current Epoch</dt>
                      <dd class="font-semibold">{job.current_epoch}</dd>
                    </div>
                  {/if}
                </div>
              </div>
            </Card>
          {/if}

          <!-- Real-time Logs -->
          {#if (job.status === "running" || job.status === "queued") && logs.length > 0}
            <Card>
              <div class="p-6">
                <div class="flex justify-between items-center mb-4">
                  <h2 class="text-xl font-semibold text-gray-900">
                    Real-time Logs
                  </h2>
                  {#if isConnected}
                    <span class="text-xs text-gray-500">Live</span>
                  {/if}
                </div>

                <div
                  bind:this={logsContainer}
                  class="bg-gray-900 text-green-400 p-4 rounded-lg overflow-auto max-h-96 text-sm font-mono"
                >
                  {#each logs as log}
                    <div class="mb-1">{log}</div>
                  {/each}
                </div>
              </div>
            </Card>
          {/if}

          <!-- Historical Logs -->
          {#if job.logs && job.logs.length > 0}
            <Card>
              <div class="p-6">
                <h2 class="text-xl font-semibold text-gray-900 mb-4">
                  Training Logs
                </h2>

                <div
                  class="bg-gray-900 text-green-400 p-4 rounded-lg overflow-auto max-h-96 text-sm font-mono"
                >
                  {#each job.logs as log}
                    <div class="mb-1">{log}</div>
                  {/each}
                </div>
              </div>
            </Card>
          {/if}
        </div>

        <!-- Sidebar -->
        <div class="space-y-6">
          <!-- Configuration -->
          <Card>
            <div class="p-6">
              <h3 class="text-lg font-semibold text-gray-900 mb-4">
                Configuration
              </h3>

              <!-- Basic Info -->
              <div class="space-y-3 text-sm mb-6">
                <div>
                  <dt class="block text-gray-700 font-medium">Model Type</dt>
                  <dd>{job.config?.model_type || job.model_type || "text"}</dd>
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
                      <dd class="text-gray-900">{job.hyperparameters.optim}</dd>
                    </div>
                    <div>
                      <dt class="text-gray-700 font-medium">LR Scheduler</dt>
                      <dd class="text-gray-900">
                        {job.hyperparameters.lr_scheduler_type || "linear"}
                      </dd>
                    </div>
                    {#if job.hyperparameters.weight_decay !== undefined}
                      <div>
                        <dt class="text-gray-700 font-medium">Weight Decay</dt>
                        <dd class="text-gray-900">
                          {job.hyperparameters.weight_decay}
                        </dd>
                      </div>
                    {/if}
                    {#if job.hyperparameters.max_grad_norm !== undefined}
                      <div>
                        <dt class="text-gray-700 font-medium">Max Grad Norm</dt>
                        <dd class="text-gray-900">
                          {job.hyperparameters.max_grad_norm}
                        </dd>
                      </div>
                    {/if}
                    {#if job.hyperparameters.eval_strategy}
                      <div>
                        <dt class="text-gray-700 font-medium">Eval Strategy</dt>
                        <dd class="text-gray-900">
                          {job.hyperparameters.eval_strategy}
                        </dd>
                      </div>
                    {/if}
                    {#if job.hyperparameters.eval_steps}
                      <div>
                        <dt class="text-gray-700 font-medium">Eval Steps</dt>
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
                          {job.lora_config.use_rslora ? "Enabled" : "Disabled"}
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
              {#if job.hyperparameters && (job.hyperparameters.adam_beta1 !== undefined || job.hyperparameters.dataloader_num_workers !== undefined || job.hyperparameters.metric_for_best_model !== undefined)}
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
                          <h5 class="text-sm font-medium text-gray-800 mb-2">
                            Optimizer Parameters
                          </h5>
                          <div class="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
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
                                <dt class="text-gray-600">Epsilon</dt>
                                <dd class="text-gray-900">
                                  {job.hyperparameters.adam_epsilon}
                                </dd>
                              </div>
                            {/if}
                          </div>
                        </div>
                      {/if}

                      <!-- Dataloader Settings -->
                      {#if job.hyperparameters.dataloader_num_workers !== undefined || job.hyperparameters.dataloader_pin_memory !== undefined}
                        <div>
                          <h5 class="text-sm font-medium text-gray-800 mb-2">
                            Dataloader Settings
                          </h5>
                          <div class="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                            {#if job.hyperparameters.dataloader_num_workers !== undefined}
                              <div>
                                <dt class="text-gray-600">Workers</dt>
                                <dd class="text-gray-900">
                                  {job.hyperparameters.dataloader_num_workers}
                                </dd>
                              </div>
                            {/if}
                            {#if job.hyperparameters.dataloader_pin_memory !== undefined}
                              <div>
                                <dt class="text-gray-600">Pin Memory</dt>
                                <dd class="text-gray-900">
                                  {job.hyperparameters.dataloader_pin_memory
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
                          <h5 class="text-sm font-medium text-gray-800 mb-2">
                            Evaluation & Saving
                          </h5>
                          <div class="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                            {#if job.hyperparameters.metric_for_best_model !== undefined}
                              <div>
                                <dt class="text-gray-600">Best Model Metric</dt>
                                <dd class="text-gray-900">
                                  {job.hyperparameters.metric_for_best_model}
                                </dd>
                              </div>
                            {/if}
                            {#if job.hyperparameters.load_best_model_at_end !== undefined}
                              <div>
                                <dt class="text-gray-600">Load Best Model</dt>
                                <dd class="text-gray-900">
                                  {job.hyperparameters.load_best_model_at_end
                                    ? "Yes"
                                    : "No"}
                                </dd>
                              </div>
                            {/if}
                            {#if job.hyperparameters.save_total_limit !== undefined}
                              <div>
                                <dt class="text-gray-600">Save Limit</dt>
                                <dd class="text-gray-900">
                                  {job.hyperparameters.save_total_limit} checkpoints
                                </dd>
                              </div>
                            {/if}
                          </div>
                        </div>
                      {/if}

                      <!-- LoRA Advanced Settings -->
                      {#if job.lora_config && (job.lora_config.use_gradient_checkpointing !== undefined || job.lora_config.random_state !== undefined)}
                        <div>
                          <h5 class="text-sm font-medium text-gray-800 mb-2">
                            LoRA Advanced
                          </h5>
                          <div class="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                            {#if job.lora_config.use_gradient_checkpointing !== undefined}
                              <div>
                                <dt class="text-gray-600">
                                  Gradient Checkpointing
                                </dt>
                                <dd class="text-gray-900">
                                  {job.lora_config.use_gradient_checkpointing}
                                </dd>
                              </div>
                            {/if}
                            {#if job.lora_config.random_state !== undefined}
                              <div>
                                <dt class="text-gray-600">Random Seed</dt>
                                <dd class="text-gray-900">
                                  {job.lora_config.random_state}
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

          <!-- Actions -->
          <Card>
            <div class="p-6">
              <h3 class="text-lg font-semibold text-gray-900 mb-4">Actions</h3>

              <div class="space-y-2">
                <Button variant="primary" fullWidth onclick={loadJob}>
                  Refresh Status
                </Button>

                {#if job.status === "running" || job.status === "queued"}
                  <Button 
                    variant="danger" 
                    fullWidth 
                    onclick={cancelJob}
                    loading={cancelling}
                    disabled={cancelling}
                    title="Mark job as cancelled (training process may continue in background)"
                  >
                    {cancelling ? "Cancelling..." : "Cancel Training"}
                  </Button>
                {/if}

                {#if job.status === "completed" && job.config}
                  <Button
                    href="/models/{job.config.name}"
                    variant="secondary"
                    fullWidth
                  >
                    View Model
                  </Button>
                {/if}

                <Button href="/training/new" variant="secondary" fullWidth>
                  Start New Job
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    {/if}
  </div>
</div>
