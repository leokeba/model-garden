<script lang="ts">
  import { api } from "$lib/api/client";
  import Badge from "$lib/components/Badge.svelte";
  import Button from "$lib/components/Button.svelte";
  import Card from "$lib/components/Card.svelte";
  import EmissionsTrendChart from "$lib/components/EmissionsTrendChart.svelte";
  import ModelComparisonChart from "$lib/components/ModelComparisonChart.svelte";
  import { onMount } from "svelte";

  type EmissionsData = {
    job_id: string;
    job_name: string;
    stage: "training" | "inference";
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

  type TrendDataPoint = {
    date: string;
    emissions_kg: number;
    energy_kwh: number;
    job_count: number;
    training_jobs: number;
    inference_jobs: number;
  };

  type ModelComparison = {
    model: string;
    emissions_kg: number;
    energy_kwh: number;
    duration_hours: number;
    job_count: number;
    avg_emissions_per_job: number;
  };

  type Recommendation = {
    id: string;
    priority: "high" | "medium" | "low" | "info";
    title: string;
    description: string;
    potential_savings_kg: number | null;
    action: string;
  };

  type Insight = {
    type: string;
    title: string;
    value: string;
    context: string;
  };

  let emissions: EmissionsData[] = $state([]);
  let loading = $state(true);
  let error = $state("");

  // Analytics state
  let activeTab = $state<
    "overview" | "trends" | "comparisons" | "recommendations" | "records"
  >("overview");
  let trendPeriod = $state<"7d" | "30d" | "90d" | "all">("30d");
  let trendData: TrendDataPoint[] = $state([]);
  let modelComparisons: ModelComparison[] = $state([]);
  let typeComparisons = $state<Record<string, any>>({});
  let recommendations: Recommendation[] = $state([]);
  let insights: Insight[] = $state([]);
  let analyticsSummary = $state<{
    total_potential_savings_kg: number;
    efficiency_score: number;
    recommendation_count: number;
  } | null>(null);
  let loadingAnalytics = $state(false);

  // Filters for records view
  let stageFilter = $state<"all" | "training" | "inference">("all");
  let sortBy = $state<"date" | "emissions" | "duration">("date");

  // Stats
  let totalEmissions = $derived(
    emissions.reduce((sum, e) => sum + e.emissions_kg, 0),
  );
  let totalEnergy = $derived(
    emissions.reduce((sum, e) => sum + e.energy_consumed, 0),
  );
  let avgCarbonIntensity = $derived(
    emissions.length > 0
      ? emissions.reduce((sum, e) => sum + e.carbon_intensity, 0) /
          emissions.length
      : 0,
  );

  // Filtered and sorted data
  let filteredEmissions = $derived(
    emissions
      .filter((e) => stageFilter === "all" || e.stage === stageFilter)
      .sort((a, b) => {
        switch (sortBy) {
          case "emissions":
            return b.emissions_kg - a.emissions_kg;
          case "duration":
            return b.duration - a.duration;
          case "date":
          default:
            return (
              new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
            );
        }
      }),
  );

  // BoAmps modal
  let selectedBoampsReport: any = $state(null);
  let loadingBoamps = $state(false);

  async function loadEmissions() {
    try {
      loading = true;
      error = "";
      const response = await api.get("/carbon/emissions");
      emissions = response.emissions || [];
    } catch (err) {
      error =
        err instanceof Error ? err.message : "Failed to load emissions data";
      emissions = [];
    } finally {
      loading = false;
    }
  }

  async function loadTrends() {
    try {
      loadingAnalytics = true;
      const response = await api.get(
        `/carbon/analytics/trends?period=${trendPeriod}&granularity=day`,
      );
      trendData = response.data_points || [];
    } catch (err) {
      console.error("Failed to load trends:", err);
      trendData = [];
    } finally {
      loadingAnalytics = false;
    }
  }

  async function loadComparisons() {
    try {
      loadingAnalytics = true;
      const response = await api.get("/carbon/analytics/comparisons");
      modelComparisons = response.by_model || [];
      typeComparisons = response.by_type || {};
    } catch (err) {
      console.error("Failed to load comparisons:", err);
      modelComparisons = [];
      typeComparisons = {};
    } finally {
      loadingAnalytics = false;
    }
  }

  async function loadRecommendations() {
    try {
      loadingAnalytics = true;
      const response = await api.get("/carbon/analytics/recommendations");
      recommendations = response.recommendations || [];
      insights = response.insights || [];
      analyticsSummary = response.summary || null;
    } catch (err) {
      console.error("Failed to load recommendations:", err);
      recommendations = [];
      insights = [];
    } finally {
      loadingAnalytics = false;
    }
  }

  async function loadBoAmpsReport(jobId: string) {
    try {
      loadingBoamps = true;
      const response = await api.get(`/carbon/boamps/${jobId}`);
      selectedBoampsReport = response;
      showBoampsModal = true;
    } catch (err) {
      error =
        err instanceof Error ? err.message : "Failed to load BoAmps report";
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
    if (kg < 0.01) return "text-green-600";
    if (kg < 0.1) return "text-yellow-600";
    return "text-red-600";
  }

  function getPriorityColor(priority: string): string {
    switch (priority) {
      case "high":
        return "bg-red-100 text-red-800 border-red-200";
      case "medium":
        return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "low":
        return "bg-blue-100 text-blue-800 border-blue-200";
      default:
        return "bg-gray-100 text-gray-800 border-gray-200";
    }
  }

  function getEfficiencyScoreColor(score: number): string {
    if (score >= 80) return "text-green-600";
    if (score >= 60) return "text-yellow-600";
    return "text-red-600";
  }

  function exportToCsv() {
    const headers = [
      "Job ID",
      "Job Name",
      "Stage",
      "Model",
      "Timestamp",
      "Duration (s)",
      "Energy (kWh)",
      "Emissions (kg CO2)",
      "Carbon Intensity (g/kWh)",
      "Country",
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

    const csv = [headers, ...rows].map((row) => row.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `emissions-report-${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Load data when tab changes
  $effect(() => {
    if (activeTab === "trends") {
      loadTrends();
    } else if (activeTab === "comparisons") {
      loadComparisons();
    } else if (activeTab === "recommendations") {
      loadRecommendations();
    }
  });

  // Reload trends when period changes
  $effect(() => {
    if (activeTab === "trends") {
      loadTrends();
    }
  });

  onMount(() => {
    loadEmissions();
    // Pre-load recommendations for overview
    loadRecommendations();
  });
</script>

<svelte:head>
  <title>Carbon Analytics - Model Garden</title>
</svelte:head>

<div class="min-h-screen bg-gray-50 pt-6">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <!-- Header -->
    <div class="flex justify-between items-center mb-8">
      <div>
        <h1 class="text-3xl font-bold text-gray-900">🌱 Carbon Analytics</h1>
        <p class="mt-2 text-sm text-gray-600">
          Track, analyze, and optimize your AI carbon footprint
        </p>
      </div>
      <Button
        onclick={exportToCsv}
        variant="secondary"
        disabled={emissions.length === 0}
      >
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
          <button
            onclick={() => (error = "")}
            class="text-red-600 hover:text-red-800"
          >
            ✕
          </button>
        </div>
      </div>
    {/if}

    {#if loading}
      <div class="flex justify-center items-center h-64">
        <div
          class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"
        ></div>
      </div>
    {:else}
      <!-- Navigation Tabs -->
      <div class="mb-6 border-b border-gray-200">
        <nav class="flex space-x-8">
          {#each [{ id: "overview", label: "📊 Overview" }, { id: "trends", label: "📈 Trends" }, { id: "comparisons", label: "⚖️ Comparisons" }, { id: "recommendations", label: "💡 Recommendations" }, { id: "records", label: "📋 Records" }] as tab}
            <button
              onclick={() => (activeTab = tab.id as any)}
              class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {activeTab ===
              tab.id
                ? 'border-primary-500 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}"
            >
              {tab.label}
            </button>
          {/each}
        </nav>
      </div>

      <!-- Overview Tab -->
      {#if activeTab === "overview"}
        <!-- Summary Stats -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card>
            <div class="p-6">
              <div class="text-sm font-medium text-gray-500 mb-1">
                Total Jobs
              </div>
              <div class="text-3xl font-bold text-gray-900">
                {emissions.length}
              </div>
            </div>
          </Card>

          <Card>
            <div class="p-6">
              <div class="text-sm font-medium text-gray-500 mb-1">
                Total Emissions
              </div>
              <div
                class="text-3xl font-bold {getEmissionsColor(totalEmissions)}"
              >
                {formatNumber(totalEmissions, 3)} kg
              </div>
              <div class="text-xs text-gray-500 mt-1">CO₂ equivalent</div>
            </div>
          </Card>

          <Card>
            <div class="p-6">
              <div class="text-sm font-medium text-gray-500 mb-1">
                Total Energy
              </div>
              <div class="text-3xl font-bold text-gray-900">
                {formatNumber(totalEnergy, 2)} kWh
              </div>
            </div>
          </Card>

          <Card>
            <div class="p-6">
              <div class="text-sm font-medium text-gray-500 mb-1">
                Avg Carbon Intensity
              </div>
              <div class="text-3xl font-bold text-gray-900">
                {formatNumber(avgCarbonIntensity, 0)}
              </div>
              <div class="text-xs text-gray-500 mt-1">g CO₂/kWh</div>
            </div>
          </Card>
        </div>

        <!-- Quick Insights & Top Recommendations -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <!-- Insights -->
          <Card>
            <div class="p-6">
              <h3 class="text-lg font-semibold text-gray-900 mb-4">
                📊 Key Insights
              </h3>
              {#if insights.length > 0}
                <div class="space-y-4">
                  {#each insights as insight}
                    <div class="p-4 bg-gray-50 rounded-lg">
                      <div class="text-sm text-gray-600">{insight.title}</div>
                      <div class="text-2xl font-bold text-gray-900">
                        {insight.value}
                      </div>
                      <div class="text-xs text-gray-500 mt-1">
                        {insight.context}
                      </div>
                    </div>
                  {/each}
                </div>
              {:else}
                <p class="text-gray-500 text-center py-8">
                  Run some jobs to see insights
                </p>
              {/if}
            </div>
          </Card>

          <!-- Top Recommendations -->
          <Card>
            <div class="p-6">
              <div class="flex justify-between items-center mb-4">
                <h3 class="text-lg font-semibold text-gray-900">
                  💡 Top Recommendations
                </h3>
                {#if analyticsSummary && analyticsSummary.total_potential_savings_kg > 0}
                  <span class="text-sm text-green-600 font-medium">
                    Potential: -{formatNumber(
                      analyticsSummary.total_potential_savings_kg,
                      3,
                    )} kg CO₂
                  </span>
                {/if}
              </div>
              {#if recommendations.length > 0}
                <div class="space-y-3">
                  {#each recommendations.slice(0, 3) as rec}
                    <div
                      class="p-4 border rounded-lg {getPriorityColor(
                        rec.priority,
                      )}"
                    >
                      <div class="flex items-start gap-3">
                        <span class="text-xl">
                          {rec.priority === "high"
                            ? "🔴"
                            : rec.priority === "medium"
                              ? "🟡"
                              : "🔵"}
                        </span>
                        <div class="flex-1">
                          <div class="font-medium">{rec.title}</div>
                          <div class="text-sm opacity-80 mt-1">
                            {rec.description}
                          </div>
                        </div>
                      </div>
                    </div>
                  {/each}
                </div>
                {#if recommendations.length > 3}
                  <button
                    onclick={() => (activeTab = "recommendations")}
                    class="mt-4 text-sm text-primary-600 hover:text-primary-800 font-medium"
                  >
                    View all {recommendations.length} recommendations →
                  </button>
                {/if}
              {:else}
                <p class="text-gray-500 text-center py-8">
                  No recommendations yet
                </p>
              {/if}
            </div>
          </Card>
        </div>

        <!-- Environmental Equivalents -->
        <Card class="mb-8">
          <div class="p-6">
            <h3 class="text-lg font-semibold text-gray-900 mb-4">
              🌍 Environmental Impact
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div class="text-center p-4 bg-green-50 rounded-lg">
                <div class="text-4xl mb-2">🚗</div>
                <div class="text-2xl font-bold text-green-700">
                  {formatNumber(totalEmissions * 4.6, 1)} km
                </div>
                <div class="text-sm text-green-600">Car equivalent</div>
              </div>
              <div class="text-center p-4 bg-blue-50 rounded-lg">
                <div class="text-4xl mb-2">📱</div>
                <div class="text-2xl font-bold text-blue-700">
                  {Math.round(totalEmissions * 121)}
                </div>
                <div class="text-sm text-blue-600">Smartphones charged</div>
              </div>
              <div class="text-center p-4 bg-emerald-50 rounded-lg">
                <div class="text-4xl mb-2">🌳</div>
                <div class="text-2xl font-bold text-emerald-700">
                  {formatNumber(totalEmissions / 0.006, 1)}
                </div>
                <div class="text-sm text-emerald-600">
                  Tree-months to offset
                </div>
              </div>
            </div>
          </div>
        </Card>
      {/if}

      <!-- Trends Tab -->
      {#if activeTab === "trends"}
        <Card class="mb-6">
          <div class="p-4 border-b border-gray-200">
            <div class="flex items-center gap-4">
              <span class="text-sm font-medium text-gray-700">Time Period:</span
              >
              <div class="flex gap-2">
                {#each [{ value: "7d", label: "7 Days" }, { value: "30d", label: "30 Days" }, { value: "90d", label: "90 Days" }, { value: "all", label: "All Time" }] as option}
                  <button
                    onclick={() => (trendPeriod = option.value as any)}
                    class="px-3 py-1 text-sm rounded-full transition-colors {trendPeriod ===
                    option.value
                      ? 'bg-primary-100 text-primary-700 font-medium'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}"
                  >
                    {option.label}
                  </button>
                {/each}
              </div>
            </div>
          </div>
          <div class="p-6">
            {#if loadingAnalytics}
              <div class="flex justify-center items-center h-64">
                <div
                  class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"
                ></div>
              </div>
            {:else}
              <EmissionsTrendChart
                dataPoints={trendData}
                title="Emissions & Energy Over Time"
                height={350}
              />
            {/if}
          </div>
        </Card>

        <!-- Trend Statistics -->
        {#if trendData.length > 0}
          <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <div class="p-6">
                <div class="text-sm font-medium text-gray-500 mb-1">
                  Period Total
                </div>
                <div class="text-2xl font-bold text-green-600">
                  {formatNumber(
                    trendData.reduce((s, d) => s + d.emissions_kg, 0),
                    4,
                  )} kg CO₂
                </div>
              </div>
            </Card>
            <Card>
              <div class="p-6">
                <div class="text-sm font-medium text-gray-500 mb-1">
                  Daily Average
                </div>
                <div class="text-2xl font-bold text-gray-900">
                  {formatNumber(
                    trendData.reduce((s, d) => s + d.emissions_kg, 0) /
                      Math.max(trendData.length, 1),
                    4,
                  )} kg
                </div>
              </div>
            </Card>
            <Card>
              <div class="p-6">
                <div class="text-sm font-medium text-gray-500 mb-1">
                  Jobs in Period
                </div>
                <div class="text-2xl font-bold text-gray-900">
                  {trendData.reduce((s, d) => s + d.job_count, 0)}
                </div>
              </div>
            </Card>
          </div>
        {/if}
      {/if}

      <!-- Comparisons Tab -->
      {#if activeTab === "comparisons"}
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <!-- By Model -->
          <Card>
            <div class="p-6">
              {#if loadingAnalytics}
                <div class="flex justify-center items-center h-64">
                  <div
                    class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"
                  ></div>
                </div>
              {:else}
                <ModelComparisonChart
                  data={modelComparisons}
                  title="CO₂ Emissions by Model"
                  height={300}
                />
              {/if}
            </div>
          </Card>

          <!-- By Type -->
          <Card>
            <div class="p-6">
              <h3 class="text-lg font-semibold text-gray-900 mb-4">
                Emissions by Job Type
              </h3>
              {#if Object.keys(typeComparisons).length > 0}
                <div class="space-y-4">
                  {#each Object.entries(typeComparisons) as [type, data]}
                    <div class="p-4 bg-gray-50 rounded-lg">
                      <div class="flex justify-between items-center mb-2">
                        <span class="font-medium text-gray-900 capitalize"
                          >{type}</span
                        >
                        <Badge
                          variant={type === "training" ? "info" : "success"}
                          >{data.job_count} jobs</Badge
                        >
                      </div>
                      <div class="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span class="text-gray-500">CO₂:</span>
                          <span class="font-medium ml-1"
                            >{formatNumber(data.emissions_kg, 4)} kg</span
                          >
                        </div>
                        <div>
                          <span class="text-gray-500">Energy:</span>
                          <span class="font-medium ml-1"
                            >{formatNumber(data.energy_kwh, 2)} kWh</span
                          >
                        </div>
                      </div>
                      <!-- Progress bar -->
                      <div
                        class="mt-3 h-2 bg-gray-200 rounded-full overflow-hidden"
                      >
                        <div
                          class="h-full {type === 'training'
                            ? 'bg-blue-500'
                            : 'bg-green-500'}"
                          style="width: {(data.emissions_kg / totalEmissions) *
                            100}%"
                        ></div>
                      </div>
                    </div>
                  {/each}
                </div>
              {:else}
                <p class="text-gray-500 text-center py-8">
                  No comparison data available
                </p>
              {/if}
            </div>
          </Card>
        </div>

        <!-- Model Statistics Table -->
        {#if modelComparisons.length > 0}
          <Card>
            <div class="p-6">
              <h3 class="text-lg font-semibold text-gray-900 mb-4">
                Model Statistics
              </h3>
              <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200">
                  <thead class="bg-gray-50">
                    <tr>
                      <th
                        class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase"
                        >Model</th
                      >
                      <th
                        class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase"
                        >Jobs</th
                      >
                      <th
                        class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase"
                        >Total CO₂</th
                      >
                      <th
                        class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase"
                        >Avg/Job</th
                      >
                      <th
                        class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase"
                        >Energy</th
                      >
                      <th
                        class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase"
                        >Duration</th
                      >
                    </tr>
                  </thead>
                  <tbody class="bg-white divide-y divide-gray-200">
                    {#each modelComparisons as model}
                      <tr class="hover:bg-gray-50">
                        <td class="px-4 py-3 text-sm font-medium text-gray-900"
                          >{model.model}</td
                        >
                        <td class="px-4 py-3 text-sm text-gray-500 text-right"
                          >{model.job_count}</td
                        >
                        <td
                          class="px-4 py-3 text-sm text-right {getEmissionsColor(
                            model.emissions_kg,
                          )}">{formatNumber(model.emissions_kg, 4)} kg</td
                        >
                        <td class="px-4 py-3 text-sm text-gray-500 text-right"
                          >{formatNumber(model.avg_emissions_per_job, 4)} kg</td
                        >
                        <td class="px-4 py-3 text-sm text-gray-500 text-right"
                          >{formatNumber(model.energy_kwh, 2)} kWh</td
                        >
                        <td class="px-4 py-3 text-sm text-gray-500 text-right"
                          >{formatNumber(model.duration_hours, 1)}h</td
                        >
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            </div>
          </Card>
        {/if}
      {/if}

      <!-- Recommendations Tab -->
      {#if activeTab === "recommendations"}
        <!-- Summary Card -->
        {#if analyticsSummary}
          <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <Card>
              <div class="p-6 text-center">
                <div class="text-4xl mb-2">⚡</div>
                <div
                  class="text-3xl font-bold {getEfficiencyScoreColor(
                    analyticsSummary.efficiency_score,
                  )}"
                >
                  {analyticsSummary.efficiency_score}/100
                </div>
                <div class="text-sm text-gray-500">Efficiency Score</div>
              </div>
            </Card>
            <Card>
              <div class="p-6 text-center">
                <div class="text-4xl mb-2">💡</div>
                <div class="text-3xl font-bold text-gray-900">
                  {analyticsSummary.recommendation_count}
                </div>
                <div class="text-sm text-gray-500">Recommendations</div>
              </div>
            </Card>
            <Card>
              <div class="p-6 text-center">
                <div class="text-4xl mb-2">🌱</div>
                <div class="text-3xl font-bold text-green-600">
                  -{formatNumber(
                    analyticsSummary.total_potential_savings_kg,
                    3,
                  )} kg
                </div>
                <div class="text-sm text-gray-500">Potential CO₂ Savings</div>
              </div>
            </Card>
          </div>
        {/if}

        <!-- Recommendations List -->
        <Card>
          <div class="p-6">
            <h3 class="text-lg font-semibold text-gray-900 mb-4">
              All Recommendations
            </h3>
            {#if loadingAnalytics}
              <div class="flex justify-center items-center h-32">
                <div
                  class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"
                ></div>
              </div>
            {:else if recommendations.length > 0}
              <div class="space-y-4">
                {#each recommendations as rec}
                  <div
                    class="p-4 border rounded-lg {getPriorityColor(
                      rec.priority,
                    )}"
                  >
                    <div class="flex items-start gap-4">
                      <div
                        class="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center
                        {rec.priority === 'high'
                          ? 'bg-red-200'
                          : rec.priority === 'medium'
                            ? 'bg-yellow-200'
                            : 'bg-blue-200'}"
                      >
                        {rec.priority === "high"
                          ? "🔴"
                          : rec.priority === "medium"
                            ? "🟡"
                            : rec.priority === "low"
                              ? "🔵"
                              : "ℹ️"}
                      </div>
                      <div class="flex-1">
                        <div class="flex items-center gap-2">
                          <h4 class="font-semibold text-gray-900">
                            {rec.title}
                          </h4>
                          <Badge
                            variant={rec.priority === "high"
                              ? "danger"
                              : rec.priority === "medium"
                                ? "warning"
                                : "info"}
                          >
                            {rec.priority}
                          </Badge>
                        </div>
                        <p class="text-sm text-gray-600 mt-1">
                          {rec.description}
                        </p>
                        <div class="flex items-center gap-4 mt-3">
                          <div class="flex items-center gap-2 text-sm">
                            <span class="text-gray-500">Action:</span>
                            <span class="font-medium text-gray-700"
                              >{rec.action}</span
                            >
                          </div>
                          {#if rec.potential_savings_kg}
                            <div class="flex items-center gap-2 text-sm">
                              <span class="text-gray-500"
                                >Potential savings:</span
                              >
                              <span class="font-medium text-green-600"
                                >-{formatNumber(rec.potential_savings_kg, 4)} kg
                                CO₂</span
                              >
                            </div>
                          {/if}
                        </div>
                      </div>
                    </div>
                  </div>
                {/each}
              </div>
            {:else}
              <div class="text-center py-12">
                <div class="text-6xl mb-4">🎉</div>
                <h3 class="text-xl font-semibold text-gray-700 mb-2">
                  You're doing great!
                </h3>
                <p class="text-gray-500">
                  No specific recommendations at this time.
                </p>
              </div>
            {/if}
          </div>
        </Card>
      {/if}

      <!-- Records Tab -->
      {#if activeTab === "records"}
        <!-- Filters and Sort -->
        <Card class="mb-6">
          <div class="p-4">
            <div class="flex flex-wrap gap-4 items-center">
              <div class="flex-1 min-w-[200px]">
                <label
                  for="stage-filter"
                  class="block text-sm font-medium text-gray-700 mb-2"
                >
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
                <label
                  for="sort-by"
                  class="block text-sm font-medium text-gray-700 mb-2"
                >
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
              <h3 class="text-xl font-semibold text-gray-700 mb-2">
                No emissions data yet
              </h3>
              <p class="text-gray-500 mb-6">
                Start training or running inference to track carbon emissions
              </p>
              <Button href="/training/new" variant="primary"
                >Start Training</Button
              >
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
                        <Badge
                          variant={emission.stage === "training"
                            ? "info"
                            : "success"}
                        >
                          {emission.stage}
                        </Badge>
                      </div>

                      <div
                        class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm"
                      >
                        <div>
                          <div class="text-gray-500">Model</div>
                          <div class="font-medium text-gray-900">
                            {emission.model_name}
                          </div>
                        </div>

                        <div>
                          <div class="text-gray-500">Emissions</div>
                          <div
                            class="font-medium {getEmissionsColor(
                              emission.emissions_kg,
                            )}"
                          >
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
                          <div class="font-medium text-gray-900">
                            {emission.country}
                          </div>
                        </div>

                        <div>
                          <div class="text-gray-500">Date</div>
                          <div
                            class="font-medium text-gray-900 truncate"
                            title={formatDate(emission.timestamp)}
                          >
                            {new Date(emission.timestamp).toLocaleDateString()}
                          </div>
                        </div>

                        <div>
                          <div class="text-gray-500">Energy Breakdown</div>
                          <div class="font-medium text-gray-900 text-xs">
                            CPU: {formatNumber(emission.cpu_energy, 2)} | GPU: {formatNumber(
                              emission.gpu_energy,
                              2,
                            )} | RAM: {formatNumber(emission.ram_energy, 2)}
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
    {/if}
  </div>
</div>

<!-- BoAmps Report Modal -->
{#if selectedBoampsReport}
  <div
    class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
  >
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
            <div
              class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"
            ></div>
          </div>
        {:else}
          <pre
            class="text-sm text-gray-900 whitespace-pre-wrap overflow-x-auto bg-gray-50 p-4 rounded-lg">{JSON.stringify(
              selectedBoampsReport,
              null,
              2,
            )}</pre>
        {/if}
      </div>

      <div class="p-6 border-t border-gray-200 flex gap-3">
        <Button
          onclick={() => {
            const blob = new Blob(
              [JSON.stringify(selectedBoampsReport, null, 2)],
              {
                type: "application/json",
              },
            );
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `boamps-report-${selectedBoampsReport.header?.reportId || "report"}.json`;
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
