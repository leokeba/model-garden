<script lang="ts">
  import { page } from "$app/stores";
  import { api, type TrainingJob } from "$lib/api/client";
  import Button from "$lib/components/Button.svelte";
  import Card from "$lib/components/Card.svelte";
  import {
    JobHeader,
    JobInfo,
    LossCurvesCard,
    ProgressCard,
    LogsCard,
    ConfigSidebar,
    CarbonCard,
    ActionsCard,
  } from "$lib/components/training-job";
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
  let cancelling = $state(false);
  let stoppingEarly = $state(false);
  let rerunning = $state(false);

  // Carbon emissions data
  let carbonData = $state<any>(null);
  let loadingCarbon = $state(false);

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

      // Load carbon emissions data
      await loadCarbonData();
    } catch (err) {
      error =
        err instanceof Error ? err.message : "Failed to load training job";
    } finally {
      loading = false;
    }
  }

  async function loadCarbonData() {
    if (!jobId) return;

    try {
      loadingCarbon = true;
      const response = await api.get(`/carbon/emissions?job_type=training`);

      // Find carbon data for this job
      if (response.emissions && Array.isArray(response.emissions)) {
        const emission = response.emissions.find(
          (e: any) => e.job_id === jobId,
        );
        carbonData = emission || null;
      }
    } catch (err) {
      console.error("Failed to load carbon data:", err);
      carbonData = null;
    } finally {
      loadingCarbon = false;
    }
  }

  function scrollLogsToBottom() {
    if (logsContainer) {
      logsContainer.scrollTop = logsContainer.scrollHeight;
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
          } else if (update.type === "early_stop_requested") {
            // Early stop was requested
            logs = [
              ...logs,
              `[${new Date().toLocaleTimeString()}] ⏸️ Early stopping requested - training will stop gracefully after current step...`,
            ];
            setTimeout(() => scrollLogsToBottom(), 10);
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
      error =
        err instanceof Error ? err.message : "Failed to cancel training job";
    } finally {
      cancelling = false;
    }
  }

  async function stopEarly() {
    if (!jobId || !job) return;

    const confirmMessage = `Stop training early for "${job.config?.name || job.name}"?\n\nThis will gracefully finish the current step and save the model.\nUnlike Cancel, early stopping lets the model save properly.`;
    if (!confirm(confirmMessage)) return;

    try {
      stoppingEarly = true;
      const response = await api.post(`/training/jobs/${jobId}/stop`, {});

      if (response.success) {
        logs = [
          ...logs,
          `[${new Date().toLocaleTimeString()}] ⏸️ Early stopping requested - training will finish current step and save...`,
        ];
        setTimeout(() => scrollLogsToBottom(), 10);
      } else {
        error = response.message || "Failed to request early stopping";
      }
    } catch (err) {
      error =
        err instanceof Error ? err.message : "Failed to request early stopping";
    } finally {
      stoppingEarly = false;
    }
  }

  async function rerunJob() {
    if (!jobId || !job) return;

    const confirmMessage = `Rerun training job "${job.config?.name || job.name}"?\n\nThis will create a new training job with the same configuration.\nThe original job will remain unchanged.`;
    if (!confirm(confirmMessage)) return;

    try {
      rerunning = true;
      const response = await api.rerunTrainingJob(jobId);

      if (response.success && response.data) {
        // Navigate to the new job
        const newJobId = response.data.job_id;
        window.location.href = `/training/${newJobId}`;
      } else {
        error = response.message || "Failed to rerun training job";
      }
    } catch (err) {
      error =
        err instanceof Error ? err.message : "Failed to rerun training job";
    } finally {
      rerunning = false;
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

<div class="min-h-screen bg-gray-50 pt-6">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
    <JobHeader
      {job}
      {isConnected}
      {cancelling}
      {stoppingEarly}
      onCancel={cancelJob}
      onStopEarly={stopEarly}
    />

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
          <JobInfo {job} />

          <LossCurvesCard {trainingMetrics} {validationMetrics} />

          <ProgressCard {job} />

          <LogsCard {job} {logs} {isConnected} bind:logsContainer />
        </div>

        <!-- Sidebar -->
        <div class="space-y-6">
          <ConfigSidebar {job} />

          <CarbonCard {carbonData} loading={loadingCarbon} />

          <ActionsCard
            {job}
            {cancelling}
            {rerunning}
            onRefresh={loadJob}
            onCancel={cancelJob}
            onRerun={rerunJob}
          />
        </div>
      </div>
    {/if}
  </div>
</div>
