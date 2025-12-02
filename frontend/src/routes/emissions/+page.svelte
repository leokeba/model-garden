<script lang="ts">
  import { api } from "$lib/api/client";
  import Button from "$lib/components/Button.svelte";
  import {
    EmissionsStats,
    EmissionsOverview,
    EmissionsTrends,
    EmissionsComparisons,
    EmissionsRecommendations,
    EmissionsRecords,
    BoAmpsModal,
  } from "$lib/components/emissions";
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

  // BoAmps modal
  let selectedBoampsReport: any = $state(null);
  let loadingBoamps = $state(false);

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
    } catch (err) {
      error =
        err instanceof Error ? err.message : "Failed to load BoAmps report";
    } finally {
      loadingBoamps = false;
    }
  }

  function exportToCsv() {
    const filteredEmissions = emissions
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
      });

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
        <EmissionsStats
          totalJobs={emissions.length}
          {totalEmissions}
          {totalEnergy}
          {avgCarbonIntensity}
        />

        <EmissionsOverview
          {insights}
          {recommendations}
          {analyticsSummary}
          {totalEmissions}
          onViewAllRecommendations={() => (activeTab = "recommendations")}
        />
      {/if}

      <!-- Trends Tab -->
      {#if activeTab === "trends"}
        <EmissionsTrends
          {trendData}
          {trendPeriod}
          loading={loadingAnalytics}
          onPeriodChange={(p) => (trendPeriod = p)}
        />
      {/if}

      <!-- Comparisons Tab -->
      {#if activeTab === "comparisons"}
        <EmissionsComparisons
          {modelComparisons}
          {typeComparisons}
          {totalEmissions}
          loading={loadingAnalytics}
        />
      {/if}

      <!-- Recommendations Tab -->
      {#if activeTab === "recommendations"}
        <EmissionsRecommendations
          {recommendations}
          {analyticsSummary}
          loading={loadingAnalytics}
        />
      {/if}

      <!-- Records Tab -->
      {#if activeTab === "records"}
        <EmissionsRecords
          {emissions}
          {stageFilter}
          {sortBy}
          onStageFilterChange={(f) => (stageFilter = f)}
          onSortByChange={(s) => (sortBy = s)}
          onRefresh={loadEmissions}
          onLoadBoAmpsReport={loadBoAmpsReport}
        />
      {/if}
    {/if}
  </div>
</div>

<!-- BoAmps Report Modal -->
<BoAmpsModal
  report={selectedBoampsReport}
  loading={loadingBoamps}
  onClose={() => (selectedBoampsReport = null)}
/>
