<script lang="ts">
  import { onMount } from 'svelte';
  import Button from '$lib/components/Button.svelte';
  import Card from '$lib/components/Card.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import { api, type TrainingJob } from '$lib/api/client';

  let jobs: TrainingJob[] = $state([]);
  let loading = $state(true);
  let error = $state('');

  onMount(async () => {
    try {
      const response = await api.getTrainingJobs();
      jobs = response.items;
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load training jobs';
    } finally {
      loading = false;
    }
  });

  function formatDate(dateString: string): string {
    return new Date(dateString).toLocaleString();
  }

  function getStatusVariant(status: string): 'success' | 'warning' | 'error' | 'info' {
    switch (status) {
      case 'completed': return 'success';
      case 'failed': case 'cancelled': return 'error';
      case 'running': return 'info';
      default: return 'warning';
    }
  }

  async function handleCancel(jobId: string) {
    if (!confirm('Are you sure you want to cancel this training job?')) return;
    
    try {
      await api.cancelTrainingJob(jobId);
      // Refresh the job list
      const response = await api.getTrainingJobs();
      jobs = response.items;
    } catch (err) {
      alert('Failed to cancel job: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  }
</script>

<svelte:head>
  <title>Training Jobs - Model Garden</title>
</svelte:head>

<div class="min-h-screen bg-gray-50">
  <!-- Header -->
  <div class="bg-white shadow">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between items-center py-6">
        <div class="flex items-center">
          <Button href="/" variant="ghost" size="sm">← Dashboard</Button>
          <h1 class="text-3xl font-bold text-gray-900 ml-4">Training Jobs</h1>
        </div>
        <div class="flex gap-3">
          <Button href="/training/new" variant="primary">
            + New Training Job
          </Button>
        </div>
      </div>
    </div>
  </div>

  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    {#if loading}
      <div class="text-center py-12">
        <div class="inline-block w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin"></div>
        <p class="mt-2 text-gray-600">Loading training jobs...</p>
      </div>
    {:else if error}
      <div class="text-center py-12">
        <div class="text-red-600 text-lg">{error}</div>
        <Button onclick={() => window.location.reload()} variant="primary" class="mt-4">
          Retry
        </Button>
      </div>
    {:else if jobs.length === 0}
      <div class="text-center py-12">
        <div class="text-gray-400 text-6xl mb-4">⚡</div>
        <h3 class="text-xl font-semibold text-gray-900 mb-2">No training jobs yet</h3>
        <p class="text-gray-500 mb-6">Start training your first model to see jobs here.</p>
        <Button href="/training/new" variant="primary">
          Start First Training Job
        </Button>
      </div>
    {:else}
      <div class="space-y-4">
        {#each jobs as job}
          <Card>
            <div class="flex items-center justify-between">
              <div class="flex-1">
                <div class="flex items-center justify-between">
                  <h3 class="text-lg font-semibold text-gray-900">{job.name}</h3>
                  <Badge variant={getStatusVariant(job.status)}>
                    {job.status}
                  </Badge>
                </div>
                
                <div class="mt-1 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-600">
                  <div>
                    <span class="font-medium">Base Model:</span> {job.base_model}
                  </div>
                  <div>
                    <span class="font-medium">Dataset:</span> {job.dataset_path}
                  </div>
                  <div>
                    <span class="font-medium">Created:</span> {formatDate(job.created_at)}
                  </div>
                </div>

                {#if job.progress && job.progress.total_steps > 0}
                  <div class="mt-3">
                    <div class="flex items-center justify-between text-sm">
                      <span class="text-gray-600">Progress</span>
                      <span class="text-gray-900">
                        {job.progress.current_step} / {job.progress.total_steps} steps
                      </span>
                    </div>
                    <div class="mt-1 w-full bg-gray-200 rounded-full h-2">
                      <div 
                        class="bg-primary-600 h-2 rounded-full transition-all"
                        style="width: {(job.progress.current_step / job.progress.total_steps) * 100}%"
                      ></div>
                    </div>
                  </div>
                {/if}

                {#if job.error_message}
                  <div class="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                    <strong>Error:</strong> {job.error_message}
                  </div>
                {/if}
              </div>

              <div class="ml-6 flex gap-2">
                <Button href="/training/{job.id}" variant="secondary" size="sm">
                  Details
                </Button>
                {#if job.status === 'running' || job.status === 'queued'}
                  <Button 
                    variant="danger" 
                    size="sm" 
                    onclick={() => handleCancel(job.id)}
                  >
                    Cancel
                  </Button>
                {/if}
              </div>
            </div>
          </Card>
        {/each}
      </div>
    {/if}
  </div>
</div>