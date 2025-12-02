<script lang="ts">
    import Badge from "$lib/components/Badge.svelte";
    import Button from "$lib/components/Button.svelte";
    import Card from "$lib/components/Card.svelte";

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

    interface Props {
        emissions: EmissionsData[];
        stageFilter: "all" | "training" | "inference";
        sortBy: "date" | "emissions" | "duration";
        onStageFilterChange: (filter: "all" | "training" | "inference") => void;
        onSortByChange: (sort: "date" | "emissions" | "duration") => void;
        onRefresh: () => void;
        onLoadBoAmpsReport: (jobId: string) => void;
    }

    let {
        emissions,
        stageFilter,
        sortBy,
        onStageFilterChange,
        onSortByChange,
        onRefresh,
        onLoadBoAmpsReport,
    }: Props = $props();

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
                            new Date(b.timestamp).getTime() -
                            new Date(a.timestamp).getTime()
                        );
                }
            }),
    );

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
</script>

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
                    value={stageFilter}
                    onchange={(e) =>
                        onStageFilterChange(
                            e.currentTarget.value as
                                | "all"
                                | "training"
                                | "inference",
                        )}
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
                    value={sortBy}
                    onchange={(e) =>
                        onSortByChange(
                            e.currentTarget.value as
                                | "date"
                                | "emissions"
                                | "duration",
                        )}
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 text-sm"
                >
                    <option value="date">Date (Newest First)</option>
                    <option value="emissions">Emissions (Highest First)</option>
                    <option value="duration">Duration (Longest First)</option>
                </select>
            </div>

            <div class="flex-shrink-0 flex items-end">
                <Button onclick={onRefresh} variant="secondary" size="sm">
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
                                        {formatNumber(emission.emissions_kg, 4)}
                                        kg CO₂
                                    </div>
                                </div>

                                <div>
                                    <div class="text-gray-500">Energy</div>
                                    <div class="font-medium text-gray-900">
                                        {formatNumber(
                                            emission.energy_consumed,
                                            3,
                                        )} kWh
                                    </div>
                                </div>

                                <div>
                                    <div class="text-gray-500">Duration</div>
                                    <div class="font-medium text-gray-900">
                                        {formatDuration(emission.duration)}
                                    </div>
                                </div>

                                <div>
                                    <div class="text-gray-500">
                                        Carbon Intensity
                                    </div>
                                    <div class="font-medium text-gray-900">
                                        {formatNumber(
                                            emission.carbon_intensity,
                                            0,
                                        )} g/kWh
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
                                        {new Date(
                                            emission.timestamp,
                                        ).toLocaleDateString()}
                                    </div>
                                </div>

                                <div>
                                    <div class="text-gray-500">
                                        Energy Breakdown
                                    </div>
                                    <div
                                        class="font-medium text-gray-900 text-xs"
                                    >
                                        CPU: {formatNumber(
                                            emission.cpu_energy,
                                            2,
                                        )} | GPU: {formatNumber(
                                            emission.gpu_energy,
                                            2,
                                        )} | RAM: {formatNumber(
                                            emission.ram_energy,
                                            2,
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="ml-4 flex flex-col gap-2">
                            {#if emission.boamps_report}
                                <Button
                                    onclick={() =>
                                        onLoadBoAmpsReport(emission.job_id)}
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
