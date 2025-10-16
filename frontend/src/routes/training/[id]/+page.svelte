<script lang="ts">
  import { page } from '$app/stores';
  import Button from '$lib/components/Button.svelte';
  import Card from '$lib/components/Card.svelte';
  import { api, type TrainingJob } from '$lib/api/client';
  import { onMount, onDestroy } from 'svelte';

  const jobId = $derived($page.params.id);
  
  let job: TrainingJob | null = $state(null);
  let loading = $state(true);
  let error = $state('');
  let ws: WebSocket | null = null;
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  let reconnectAttempts = $state(0);
  let maxReconnectAttempts = 5;
  let isConnected = $state(false);
  let logs: string[] = $state([]);
  let logsContainer = $state<HTMLDivElement | null>(null);

  // Get WebSocket URL dynamically
  function getWebSocketUrl(jobId: string): string {
    if (typeof window === 'undefined') return '';
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/ws/training/${jobId}`;
  }

  async function loadJob() {
    if (!jobId) return;
    
    try {
      loading = true;
      const response = await api.getTrainingJob(jobId);
      job = response;
      error = '';
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load training job';
    } finally {
      loading = false;
    }
  }

  function connectWebSocket() {
    if (!jobId || typeof window === 'undefined') return;
    
    // Don't reconnect if job is completed or failed
    if (job && (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled')) {
      return;
    }

    try {
      const wsUrl = getWebSocketUrl(jobId);
      console.log(`Connecting to WebSocket: ${wsUrl}`);
      
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('WebSocket connected');
        isConnected = true;
        reconnectAttempts = 0;
      };

      ws.onmessage = (event) => {
        try {
          const update = JSON.parse(event.data);
          console.log('WebSocket update:', update);
          
          // Handle different update types
          if (update.type === 'status' && job) {
            job.status = update.status;
            if (update.completed_at) {
              job.completed_at = update.completed_at;
            }
          } else if (update.type === 'progress' && job) {
            job.progress = update.progress;
            job.current_step = update.progress?.current_step;
            job.total_steps = update.progress?.total_steps;
            job.current_epoch = update.progress?.epoch;
          } else if (update.type === 'log') {
            logs = [...logs, `[${new Date().toLocaleTimeString()}] ${update.message}`];
            // Keep only last 100 log lines
            if (logs.length > 100) {
              logs = logs.slice(-100);
            }
            // Auto-scroll to bottom
            setTimeout(() => scrollLogsToBottom(), 10);
          } else if (update.type === 'error' && job) {
            job.error_message = update.message;
            error = update.message;
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        isConnected = false;
      };

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        isConnected = false;
        ws = null;

        // Attempt to reconnect if job is still running
        if (job && job.status === 'running' && reconnectAttempts < maxReconnectAttempts) {
          reconnectAttempts++;
          console.log(`Reconnecting... (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
          reconnectTimeout = setTimeout(() => {
            connectWebSocket();
          }, Math.min(1000 * Math.pow(2, reconnectAttempts), 30000)); // Exponential backoff, max 30s
        }
      };
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
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
      case 'running': return 'text-blue-600 bg-blue-100';
      case 'completed': return 'text-green-600 bg-green-100';
      case 'failed': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  }

  function formatProgress(progress: any) {
    if (typeof progress === 'number') {
      return Math.round(progress * 100);
    }
    if (progress && typeof progress.current_step === 'number' && typeof progress.total_steps === 'number') {
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

  onMount(async () => {
    await loadJob();
    
    // Connect WebSocket if job is running or queued
    if (job && (job.status === 'running' || job.status === 'queued')) {
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
          <Button href="/training" variant="ghost" size="sm">← Training Jobs</Button>
          <h1 class="text-3xl font-bold text-gray-900 ml-4">Training Job Details</h1>
        </div>
        {#if job}
          <div class="flex items-center gap-2">
            {#if job.status === 'running'}
              {#if isConnected}
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-green-600 bg-green-100">
                  <span class="w-2 h-2 bg-green-600 rounded-full mr-1.5 animate-pulse"></span>
                  Live Updates
                </span>
              {:else}
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-yellow-600 bg-yellow-100">
                  <span class="w-2 h-2 bg-yellow-600 rounded-full mr-1.5"></span>
                  Reconnecting...
                </span>
              {/if}
            {/if}
            <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium {getStatusColor(job.status)}">
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
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    {:else if error}
      <Card>
        <div class="text-center py-8">
          <p class="text-red-600 text-lg">{error}</p>
          <Button onclick={loadJob} variant="primary" class="mt-4">Try Again</Button>
        </div>
      </Card>
    {:else if job}
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Main Job Info -->
        <div class="lg:col-span-2 space-y-6">
          <Card>
            <div class="p-6">
              <h2 class="text-xl font-semibold text-gray-900 mb-4">Job Information</h2>
              
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <dt class="text-sm font-medium text-gray-700">Job ID</dt>
                  <dd class="mt-1 text-sm text-gray-900">{job.job_id || job.id}</dd>
                </div>
                
                <div>
                  <dt class="text-sm font-medium text-gray-700">Status</dt>
                  <dd class="mt-1">
                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium {getStatusColor(job.status)}">
                      {job.status}
                    </span>
                  </dd>
                </div>
                
                <div>
                  <dt class="text-sm font-medium text-gray-700">Model Name</dt>
                  <dd class="mt-1 text-sm text-gray-900">{job.config?.name || job.name}</dd>
                </div>
                
                <div>
                  <dt class="text-sm font-medium text-gray-700">Base Model</dt>
                  <dd class="mt-1 text-sm text-gray-900">{job.config?.base_model || job.base_model}</dd>
                </div>
                
                <div>
                  <dt class="text-sm font-medium text-gray-700">Dataset</dt>
                  <dd class="mt-1 text-sm text-gray-900">{job.config?.dataset_path || job.dataset_path}</dd>
                </div>
                
                <div>
                  <dt class="text-sm font-medium text-gray-700">Output Directory</dt>
                  <dd class="mt-1 text-sm text-gray-900">{job.config?.output_dir || job.output_dir}</dd>
                </div>
                
                <div>
                  <dt class="text-sm font-medium text-gray-700">Created</dt>
                  <dd class="mt-1 text-sm text-gray-900">{formatDate(job.created_at)}</dd>
                </div>
                
                {#if job.completed_at}
                  <div>
                    <dt class="text-sm font-medium text-gray-700">Completed</dt>
                    <dd class="mt-1 text-sm text-gray-900">{formatDate(job.completed_at)}</dd>
                  </div>
                {/if}
              </div>
            </div>
          </Card>

          <!-- Progress -->
          {#if job.status === 'running' && job.progress}
            <Card>
              <div class="p-6">
                <h2 class="text-xl font-semibold text-gray-900 mb-4">Training Progress</h2>
                
                <div class="space-y-4">
                  <div>
                    <div class="flex justify-between text-sm text-gray-700 mb-1">
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
          {#if (job.status === 'running' || job.status === 'queued') && logs.length > 0}
            <Card>
              <div class="p-6">
                <div class="flex justify-between items-center mb-4">
                  <h2 class="text-xl font-semibold text-gray-900">Real-time Logs</h2>
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
                <h2 class="text-xl font-semibold text-gray-900 mb-4">Training Logs</h2>
                
                <div class="bg-gray-900 text-green-400 p-4 rounded-lg overflow-auto max-h-96 text-sm font-mono">
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
              <h3 class="text-lg font-semibold text-gray-900 mb-4">Configuration</h3>
              
              <div class="space-y-3 text-sm">
                <div>
                  <dt class="block text-gray-700 font-medium">Learning Rate</dt>
                  <dd>{job.config?.hyperparameters?.learning_rate || job.hyperparameters?.learning_rate}</dd>
                </div>
                
                <div>
                  <dt class="block text-gray-700 font-medium">Epochs</dt>
                  <dd>{job.config?.hyperparameters?.num_epochs || job.hyperparameters?.num_epochs}</dd>
                </div>
                
                <div>
                  <dt class="block text-gray-700 font-medium">Batch Size</dt>
                  <dd>{job.config?.hyperparameters?.batch_size || job.hyperparameters?.batch_size}</dd>
                </div>
                
                <div>
                  <dt class="block text-gray-700 font-medium">LoRA Rank</dt>
                  <dd>{job.config?.lora_config?.r || job.lora_config?.r}</dd>
                </div>
                
                <div>
                  <dt class="block text-gray-700 font-medium">LoRA Alpha</dt>
                  <dd>{job.config?.lora_config?.lora_alpha || job.lora_config?.lora_alpha}</dd>
                </div>
              </div>
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
                
                {#if job.status === 'completed' && job.config}
                  <Button href="/models/{job.config.name}" variant="secondary" fullWidth>
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