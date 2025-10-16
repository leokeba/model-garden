<script lang="ts">
  import { onMount } from 'svelte';
  import Button from '$lib/components/Button.svelte';
  import Card from '$lib/components/Card.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import { api } from '$lib/api/client';

  type EmissionsData = {
    job_id: string;
    job_name: string;
    stage: 'training' | 'inference';
    model_name: string;
    timestamp: string;
    duration: number;
    energy_consumed: number;
    emissions_kg: number;
    emissions_rate: number;
    cpu_energy: number;
    gpu_energy: number;
    ram_energy: number;
    carbon_intensity: number;
    country: string;
    boamps_report?: string;
  };

  let emissions: EmissionsData[] = $state([]);
  let loading = $state(true);
  let error = $state('');
  
  // Filters
  let stageFilter = $state<'all' | 'training' | 'inference'>('all');
  let sortBy = $state<'date' | 'emissions' | 'duration'>('date');

  // Stats
  let totalEmissions = $derived(
    emissions.reduce((sum, e) => sum + e.emissions_kg, 0)
  );
  let totalEnergy = $derived(
    emissions.reduce((sum, e) => sum + e.energy_consumed, 0)
  );
  let avgCarbonIntensity = $derived(
    emissions.length > 0
      ? emissions.reduce((sum, e) => sum + e.carbon_intensity, 0) / emissions.length
      : 0
  );

  // Filtered and sorted data
  let filteredEmissions = $derived(
    emissions
      .filter((e) => stageFilter === 'all' || e.stage === stageFilter)
      .sort((a, b) => {
        switch (sortBy) {
          case 'emissions':
            return b.emissions_kg - a.emissions_kg;
          case 'duration':
            return b.duration - a.duration;
          case 'date':
          default:
            return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
        }
      })
  );

  // BoAmps modal
  let selectedBoampsReport: any = $state(null);
  let loadingBoamps = $state(false);
  let showBoampsModal = $state(false);

  async function loadEmissions() {
    try {
      loading = true;
      error = '';
      const response = await api.get('/carbon/emissions');
      emissions = response.emissions || [];
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load emissions data';
      emissions = [];
    } finally {
      loading = false;
    }
  }
  
  async function loadBoAmpsReport(jobId: string) {
    try {
      loadingBoamps = true;
      const response = await api.get(`/carbon/boamps/${jobId}`);
      selectedBoampsReport = response;
      showBoampsModal = true;
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load BoAmps report';
    } finally {
      loadingBoamps = false;
    }
  }

  function formatNumber(num: number, decimals = 2): string {
    return num.toFixed(decimals);
  }

  function formatDuration(seconds: number): string {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
  }

  function formatDate(dateString: string): string {
    return new Date(dateString).toLocaleString();
  }

  function getEmissionsColor(kg: number): string {
    if (kg < 0.01) return 'text-green-600';
    if (kg < 0.1) return 'text-yellow-600';
    return 'text-red-600';
  }

  function exportToCsv() {
    const headers = [
      'Job ID',
      'Job Name',
      'Stage',
      'Model',
      'Timestamp',
      'Duration (s)',
      'Energy (kWh)',
      'Emissions (kg CO2)',
      'Carbon Intensity (g/kWh)',
      'Country',
    ];

    const rows = filteredEmissions.map((e) => [
      e.job_id,
      e.job_name,
      e.stage,
      e.model_name,
      e.timestamp,
      e.duration.toString(),
      e.energy_consumed.toString(),
      e.emissions_kg.toString(),
      e.carbon_intensity.toString(),
      e.country,
    ]);

    const csv = [headers, ...rows].map((row) => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `emissions-report-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  onMount(() => {
    loadEmissions();
  });
</script>

<svelte:head>
  <title>Emissions - Model Garden</title>
</svelte:head>

<div class="min-h-screen bg-gray-50 pt-6">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <!-- Header -->
    <div class="flex justify-between items-center mb-8">
      <div>
        <h1 class="text-3xl font-bold text-gray-900">🌱 Carbon Emissions</h1>
        <p class="mt-2 text-sm text-gray-600">
          Track and analyze the carbon footprint of your AI models
        </p>
      </div>
      <Button onclick={exportToCsv} variant="secondary" disabled={emissions.length === 0}>
        📥 Export CSV
      </Button>
    </div>

    {#if error}
      <div class="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
        <div class="flex items-start">
          <span class="text-red-600 mr-2">⚠️</span>
          <div class="flex-1">
            <p class="text-sm text-red-800">{error}</p>
          </div>
          <button onclick={() => (error = '')} class="text-red-600 hover:text-red-800">
            ✕
          </button>
        </div>
      </div>
    {/if}

    {#if loading}
      <div class="flex justify-center items-center h-64">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    {:else}
      <!-- Summary Stats -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <Card>
          <div class="p-6">
            <div class="text-sm font-medium text-gray-500 mb-1">Total Jobs</div>
            <div class="text-3xl font-bold text-gray-900">{emissions.length}</div>
          </div>
        </Card>

        <Card>
          <div class="p-6">
            <div class="text-sm font-medium text-gray-500 mb-1">Total Emissions</div>
            <div class="text-3xl font-bold {getEmissionsColor(totalEmissions)}">
              {formatNumber(totalEmissions, 3)} kg
            </div>
            <div class="text-xs text-gray-500 mt-1">CO₂ equivalent</div>
          </div>
        </Card>

        <Card>
          <div class="p-6">
            <div class="text-sm font-medium text-gray-500 mb-1">Total Energy</div>
            <div class="text-3xl font-bold text-gray-900">
              {formatNumber(totalEnergy, 2)} kWh
            </div>
          </div>
        </Card>

        <Card>
          <div class="p-6">
            <div class="text-sm font-medium text-gray-500 mb-1">Avg Carbon Intensity</div>
            <div class="text-3xl font-bold text-gray-900">
              {formatNumber(avgCarbonIntensity, 0)}
            </div>
            <div class="text-xs text-gray-500 mt-1">g CO₂/kWh</div>
          </div>
        </Card>
      </div>

      <!-- Filters and Sort -->
      <Card class="mb-6">
        <div class="p-4">
          <div class="flex flex-wrap gap-4 items-center">
            <div class="flex-1 min-w-[200px]">
              <label for="stage-filter" class="block text-sm font-medium text-gray-700 mb-2">
                Filter by Stage
              </label>
              <select
                id="stage-filter"
                bind:value={stageFilter}
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 text-sm"
              >
                <option value="all">All Stages</option>
                <option value="training">Training</option>
                <option value="inference">Inference</option>
              </select>
            </div>

            <div class="flex-1 min-w-[200px]">
              <label for="sort-by" class="block text-sm font-medium text-gray-700 mb-2">
                Sort By
              </label>
              <select
                id="sort-by"
                bind:value={sortBy}
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 text-sm"
              >
                <option value="date">Date (Newest First)</option>
                <option value="emissions">Emissions (Highest First)</option>
                <option value="duration">Duration (Longest First)</option>
              </select>
            </div>

            <div class="flex-shrink-0 flex items-end">
              <Button onclick={loadEmissions} variant="secondary" size="sm">
                🔄 Refresh
              </Button>
            </div>
          </div>
        </div>
      </Card>

      <!-- Emissions List -->
      {#if filteredEmissions.length === 0}
        <Card>
          <div class="text-center py-12">
            <div class="text-6xl mb-4">🌱</div>
            <h3 class="text-xl font-semibold text-gray-700 mb-2">No emissions data yet</h3>
            <p class="text-gray-500 mb-6">
              Start training or running inference to track carbon emissions
            </p>
            <Button href="/training/new" variant="primary">Start Training</Button>
          </div>
        </Card>
      {:else}
        <div class="space-y-4">
          {#each filteredEmissions as emission}
            <Card class="hover:shadow-md transition-shadow">
              <div class="p-6">
                <div class="flex items-start justify-between">
                  <div class="flex-1">
                    <div class="flex items-center gap-3 mb-2">
                      <h3 class="text-lg font-semibold text-gray-900">
                        {emission.job_name || emission.job_id}
                      </h3>
                      <Badge variant={emission.stage === 'training' ? 'info' : 'success'}>
                        {emission.stage}
                      </Badge>
                    </div>

                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <div class="text-gray-500">Model</div>
                        <div class="font-medium text-gray-900">{emission.model_name}</div>
                      </div>

                      <div>
                        <div class="text-gray-500">Emissions</div>
                        <div class="font-medium {getEmissionsColor(emission.emissions_kg)}">
                          {formatNumber(emission.emissions_kg, 4)} kg CO₂
                        </div>
                      </div>

                      <div>
                        <div class="text-gray-500">Energy</div>
                        <div class="font-medium text-gray-900">
                          {formatNumber(emission.energy_consumed, 3)} kWh
                        </div>
                      </div>

                      <div>
                        <div class="text-gray-500">Duration</div>
                        <div class="font-medium text-gray-900">
                          {formatDuration(emission.duration)}
                        </div>
                      </div>

                      <div>
                        <div class="text-gray-500">Carbon Intensity</div>
                        <div class="font-medium text-gray-900">
                          {formatNumber(emission.carbon_intensity, 0)} g/kWh
                        </div>
                      </div>

                      <div>
                        <div class="text-gray-500">Location</div>
                        <div class="font-medium text-gray-900">{emission.country}</div>
                      </div>

                      <div>
                        <div class="text-gray-500">Date</div>
                        <div class="font-medium text-gray-900 truncate" title={formatDate(emission.timestamp)}>
                          {new Date(emission.timestamp).toLocaleDateString()}
                        </div>
                      </div>

                      <div>
                        <div class="text-gray-500">Energy Breakdown</div>
                        <div class="font-medium text-gray-900 text-xs">
                          CPU: {formatNumber(emission.cpu_energy, 2)} | 
                          GPU: {formatNumber(emission.gpu_energy, 2)} | 
                          RAM: {formatNumber(emission.ram_energy, 2)}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div class="ml-4 flex flex-col gap-2">
                    {#if emission.boamps_report}
                      <Button
                        onclick={() => loadBoAmpsReport(emission.job_id)}
                        variant="secondary"
                        size="sm"
                      >
                        📄 BoAmps
                      </Button>
                    {/if}
                    <Button
                      href={`/training/${emission.job_id}`}
                      variant="ghost"
                      size="sm"
                    >
                      View Job
                    </Button>
                  </div>
                </div>
              </div>
            </Card>
          {/each}
        </div>
      {/if}
    {/if}
  </div>
</div>

<!-- BoAmps Report Modal -->
{#if selectedBoampsReport}
  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
    <Card class="max-w-4xl w-full max-h-[80vh] flex flex-col">
      <div class="p-6 border-b border-gray-200">
        <div class="flex justify-between items-center">
          <div>
            <h2 class="text-2xl font-bold text-gray-900">BoAmps Report</h2>
            <p class="text-sm text-gray-600 mt-1">
              Standardized emissions report
            </p>
          </div>
          <button
            onclick={() => {
              selectedBoampsReport = null;
            }}
            class="text-gray-400 hover:text-gray-600"
          >
            ✕
          </button>
        </div>
      </div>

      <div class="flex-1 overflow-y-auto p-6">
        {#if loadingBoamps}
          <div class="flex justify-center items-center h-32">
            <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
          </div>
        {:else}
          <pre class="text-sm text-gray-900 whitespace-pre-wrap overflow-x-auto bg-gray-50 p-4 rounded-lg">{JSON.stringify(
              selectedBoampsReport,
              null,
              2
            )}</pre>
        {/if}
      </div>

      <div class="p-6 border-t border-gray-200 flex gap-3">
        <Button
          onclick={() => {
            const blob = new Blob([JSON.stringify(selectedBoampsReport, null, 2)], {
              type: 'application/json',
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `boamps-report-${selectedBoampsReport.header?.reportId || 'report'}.json`;
            a.click();
            URL.revokeObjectURL(url);
          }}
          variant="secondary"
        >
          📥 Download JSON
        </Button>
        <Button
          onclick={() => {
            selectedBoampsReport = null;
          }}
          variant="secondary"
          fullWidth
        >
          Close
        </Button>
      </div>
    </Card>
  </div>
{/if}
