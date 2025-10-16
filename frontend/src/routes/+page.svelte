<script lang="ts">
  import { onMount } from 'svelte';
  import Button from '$lib/components/Button.svelte';
  import Card from '$lib/components/Card.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import { api, type Model, type TrainingJob, type SystemStatus } from '$lib/api/client';

  let models: Model[] = $state([]);
  let recentJobs: TrainingJob[] = $state([]);
  let systemStatus: SystemStatus | null = $state(null);
  let loading = $state(true);
  let error = $state('');

  onMount(async () => {
    try {
      // Load data in parallel
      const [modelsResponse, jobsResponse, statusResponse] = await Promise.all([
        api.getModels(),
        api.getTrainingJobs(),
        api.getSystemStatus()
      ]);

      models = modelsResponse.items;
      recentJobs = jobsResponse.items.slice(0, 5); // Recent 5 jobs
      systemStatus = statusResponse;
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load data';
    } finally {
      loading = false;
    }
  });

  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function formatDate(dateString: string): string {
    if (dateString === 'unknown') return 'Unknown';
    return new Date(dateString).toLocaleDateString();
  }
</script>

<svelte:head>
  <title>Model Garden - Dashboard</title>
</svelte:head>

<div class="min-h-screen bg-gray-50">
  <!-- Header -->
  <div class="bg-white shadow">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between items-center py-6">
        <div class="flex items-center">
          <h1 class="text-3xl font-bold text-gray-900">🌱 Model Garden</h1>
          <Badge variant="info" size="sm" class="ml-3">Dashboard</Badge>
        </div>
        <div class="flex gap-3">
          <Button href="/training/new" variant="primary">
            + New Training
          </Button>
          <Button href="/models" variant="secondary">
            View Models
          </Button>
        </div>
      </div>
    </div>
  </div>

  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    {#if loading}
      <div class="text-center py-12">
        <div class="inline-block w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin"></div>
        <p class="mt-2 text-gray-600">Loading dashboard...</p>
      </div>
    {:else if error}
      <div class="text-center py-12">
        <div class="text-red-600 text-lg">{error}</div>
        <Button onclick={() => window.location.reload()} variant="primary" class="mt-4">
          Retry
        </Button>
      </div>
    {:else}
      <!-- Stats Cards -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <Card>
          <div class="flex items-center">
            <div class="flex-shrink-0">
              <div class="w-8 h-8 bg-primary-100 rounded-lg flex items-center justify-center">
                <span class="text-primary-600 text-lg">📦</span>
              </div>
            </div>
            <div class="ml-4">
              <div class="text-sm font-medium text-gray-500">Models</div>
              <div class="text-2xl font-bold text-gray-900">{systemStatus?.storage.models_count || 0}</div>
            </div>
          </div>
        </Card>

        <Card>
          <div class="flex items-center">
            <div class="flex-shrink-0">
              <div class="w-8 h-8 bg-carbon-100 rounded-lg flex items-center justify-center">
                <span class="text-carbon-600 text-lg">⚡</span>
              </div>
            </div>
            <div class="ml-4">
              <div class="text-sm font-medium text-gray-500">Training Jobs</div>
              <div class="text-2xl font-bold text-gray-900">{systemStatus?.storage.training_jobs_count || 0}</div>
            </div>
          </div>
        </Card>

        <Card>
          <div class="flex items-center">
            <div class="flex-shrink-0">
              <div class="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                <span class="text-green-600 text-lg">🚀</span>
              </div>
            </div>
            <div class="ml-4">
              <div class="text-sm font-medium text-gray-500">Active Jobs</div>
              <div class="text-2xl font-bold text-gray-900">{systemStatus?.storage.active_jobs || 0}</div>
            </div>
          </div>
        </Card>
      </div>

      <!-- System Status -->
      {#if systemStatus}
        <div class="mb-8">
          <Card>
            <h3 class="text-lg font-semibold text-gray-900 mb-4">System Status</h3>
            
            <!-- System Metrics -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <div>
                <div class="text-sm text-gray-500">CPU Cores</div>
                <div class="font-medium">{systemStatus.system.cpu_count}</div>
                {#if systemStatus.system.cpu_percent}
                  <div class="text-xs text-gray-400">{systemStatus.system.cpu_percent.toFixed(1)}% used</div>
                {/if}
              </div>
              <div>
                <div class="text-sm text-gray-500">System Memory</div>
                <div class="font-medium">
                  {formatBytes(systemStatus.system.memory_used)} / {formatBytes(systemStatus.system.memory_total)}
                </div>
                {#if systemStatus.system.memory_percent}
                  <div class="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                    <div class="bg-blue-600 h-1.5 rounded-full" style="width: {systemStatus.system.memory_percent}%"></div>
                  </div>
                  <div class="text-xs text-gray-400 mt-0.5">{systemStatus.system.memory_percent.toFixed(1)}% used</div>
                {/if}
              </div>
              <div>
                <div class="text-sm text-gray-500">GPU Status</div>
                <div class="font-medium">
                  {#if systemStatus.gpu.available}
                    <Badge variant="success" size="sm">
                      {systemStatus.gpu.device_count || 0} GPU{(systemStatus.gpu.device_count || 0) > 1 ? 's' : ''}
                    </Badge>
                  {:else}
                    <Badge variant="error" size="sm">Not Available</Badge>
                  {/if}
                </div>
              </div>
              <div>
                <div class="text-sm text-gray-500">Disk Space</div>
                <div class="font-medium">
                  {formatBytes(systemStatus.system.disk_usage.free)} free
                </div>
                {#if systemStatus.system.disk_usage.percent}
                  <div class="text-xs text-gray-400">{systemStatus.system.disk_usage.percent.toFixed(1)}% used</div>
                {/if}
              </div>
            </div>

            <!-- GPU Details -->
            {#if systemStatus.gpu.available && systemStatus.gpu.devices}
              <div class="border-t pt-4">
                <h4 class="text-sm font-semibold text-gray-700 mb-3">GPU Details</h4>
                <div class="space-y-4">
                  {#each systemStatus.gpu.devices as gpu}
                    <div class="bg-gray-50 rounded-lg p-4">
                      <div class="flex items-center justify-between mb-3">
                        <div class="flex items-center gap-2">
                          <span class="text-lg">🎮</span>
                          <div>
                            <div class="font-medium text-gray-900">{gpu.name}</div>
                            <div class="text-xs text-gray-500">GPU {gpu.id}</div>
                          </div>
                        </div>
                        {#if gpu.temperature}
                          <div class="text-sm">
                            <span class="text-gray-500">Temp:</span>
                            <span class="font-medium" class:text-orange-600={gpu.temperature > 80} class:text-yellow-600={gpu.temperature > 70 && gpu.temperature <= 80} class:text-green-600={gpu.temperature <= 70}>
                              {gpu.temperature}°C
                            </span>
                          </div>
                        {/if}
                      </div>
                      
                      <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <!-- VRAM Usage -->
                        <div>
                          <div class="flex justify-between text-xs text-gray-600 mb-1">
                            <span>VRAM</span>
                            <span>{formatBytes(gpu.memory.used)} / {formatBytes(gpu.memory.total)}</span>
                          </div>
                          <div class="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              class="h-2 rounded-full transition-all"
                              class:bg-red-500={gpu.memory.used_percent > 90}
                              class:bg-yellow-500={gpu.memory.used_percent > 70 && gpu.memory.used_percent <= 90}
                              class:bg-green-500={gpu.memory.used_percent <= 70}
                              style="width: {gpu.memory.used_percent}%"
                            ></div>
                          </div>
                          <div class="text-xs text-gray-500 mt-0.5">{gpu.memory.used_percent}% used</div>
                        </div>

                        <!-- GPU Utilization -->
                        {#if gpu.utilization.gpu !== null}
                          <div>
                            <div class="flex justify-between text-xs text-gray-600 mb-1">
                              <span>GPU Utilization</span>
                              <span>{gpu.utilization.gpu}%</span>
                            </div>
                            <div class="w-full bg-gray-200 rounded-full h-2">
                              <div 
                                class="bg-blue-500 h-2 rounded-full transition-all"
                                style="width: {gpu.utilization.gpu}%"
                              ></div>
                            </div>
                          </div>
                        {/if}
                      </div>

                      <!-- Power Usage -->
                      {#if gpu.power}
                        <div class="mt-3 text-xs text-gray-600">
                          <span>Power:</span>
                          <span class="font-medium text-gray-900">{gpu.power.usage.toFixed(1)}W</span>
                          <span class="text-gray-400">/ {gpu.power.limit.toFixed(0)}W</span>
                        </div>
                      {/if}
                    </div>
                  {/each}
                </div>
              </div>
            {/if}
          </Card>
        </div>
      {/if}

      <!-- Recent Models and Jobs -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <!-- Recent Models -->
        <div>
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-xl font-semibold text-gray-900">Recent Models</h2>
            <Button href="/models" variant="ghost" size="sm">View All</Button>
          </div>
          
          {#if models.length === 0}
            <Card>
              <div class="text-center py-8">
                <div class="text-gray-400 text-4xl mb-2">📦</div>
                <p class="text-gray-500">No models yet</p>
                <Button href="/training/new" variant="primary" size="sm" class="mt-3">
                  Train Your First Model
                </Button>
              </div>
            </Card>
          {:else}
            <div class="space-y-3">
              {#each models.slice(0, 3) as model}
                <Card hoverable>
                  <div class="flex items-center justify-between">
                    <div>
                      <h4 class="font-medium text-gray-900">{model.name}</h4>
                      <p class="text-sm text-gray-500">{model.base_model}</p>
                      {#if model.size_bytes}
                        <p class="text-xs text-gray-400">{formatBytes(model.size_bytes)}</p>
                      {/if}
                    </div>
                    <div class="text-right">
                      <Badge variant={model.status === 'available' ? 'success' : 'warning'} size="sm">
                        {model.status}
                      </Badge>
                      <p class="text-xs text-gray-400 mt-1">{formatDate(model.created_at)}</p>
                    </div>
                  </div>
                </Card>
              {/each}
            </div>
          {/if}
        </div>

        <!-- Recent Training Jobs -->
        <div>
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-xl font-semibold text-gray-900">Recent Training Jobs</h2>
            <Button href="/training" variant="ghost" size="sm">View All</Button>
          </div>
          
          {#if recentJobs.length === 0}
            <Card>
              <div class="text-center py-8">
                <div class="text-gray-400 text-4xl mb-2">⚡</div>
                <p class="text-gray-500">No training jobs yet</p>
                <Button href="/training/new" variant="primary" size="sm" class="mt-3">
                  Start Training
                </Button>
              </div>
            </Card>
          {:else}
            <div class="space-y-3">
              {#each recentJobs as job}
                <Card hoverable>
                  <div class="flex items-center justify-between">
                    <div>
                      <h4 class="font-medium text-gray-900">{job.name}</h4>
                      <p class="text-sm text-gray-500">{job.base_model}</p>
                      {#if job.progress}
                        <div class="text-xs text-gray-400">
                          Step {job.progress.current_step} / {job.progress.total_steps}
                        </div>
                      {/if}
                    </div>
                    <div class="text-right">
                      <Badge 
                        variant={
                          job.status === 'completed' ? 'success' :
                          job.status === 'failed' ? 'error' :
                          job.status === 'running' ? 'info' : 'warning'
                        } 
                        size="sm"
                      >
                        {job.status}
                      </Badge>
                      <p class="text-xs text-gray-400 mt-1">{formatDate(job.created_at)}</p>
                    </div>
                  </div>
                </Card>
              {/each}
            </div>
          {/if}
        </div>
      </div>
    {/if}
  </div>
</div>
