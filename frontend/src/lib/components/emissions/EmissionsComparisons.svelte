<script lang="ts">
    import Badge from "$lib/components/Badge.svelte";
    import Card from "$lib/components/Card.svelte";
    import ModelComparisonChart from "$lib/components/ModelComparisonChart.svelte";

    interface ModelComparison {
        model: string;
        emissions_kg: number;
        energy_kwh: number;
        duration_hours: number;
        job_count: number;
        avg_emissions_per_job: number;
    }

    interface Props {
        modelComparisons: ModelComparison[];
        typeComparisons: Record<string, any>;
        totalEmissions: number;
        loading: boolean;
    }

    let { modelComparisons, typeComparisons, totalEmissions, loading }: Props =
        $props();

    function formatNumber(num: number, decimals = 2): string {
        return num.toFixed(decimals);
    }

    function getEmissionsColor(kg: number): string {
        if (kg < 0.01) return "text-green-600";
        if (kg < 0.1) return "text-yellow-600";
        return "text-red-600";
    }
</script>

<div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
    <!-- By Model -->
    <Card>
        <div class="p-6">
            {#if loading}
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
                                <span
                                    class="font-medium text-gray-900 capitalize"
                                    >{type}</span
                                >
                                <Badge
                                    variant={type === "training"
                                        ? "info"
                                        : "success"}
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
                                    style="width: {(data.emissions_kg /
                                        totalEmissions) *
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
                                <td
                                    class="px-4 py-3 text-sm font-medium text-gray-900"
                                    >{model.model}</td
                                >
                                <td
                                    class="px-4 py-3 text-sm text-gray-500 text-right"
                                    >{model.job_count}</td
                                >
                                <td
                                    class="px-4 py-3 text-sm text-right {getEmissionsColor(
                                        model.emissions_kg,
                                    )}"
                                    >{formatNumber(model.emissions_kg, 4)} kg</td
                                >
                                <td
                                    class="px-4 py-3 text-sm text-gray-500 text-right"
                                    >{formatNumber(
                                        model.avg_emissions_per_job,
                                        4,
                                    )} kg</td
                                >
                                <td
                                    class="px-4 py-3 text-sm text-gray-500 text-right"
                                    >{formatNumber(model.energy_kwh, 2)} kWh</td
                                >
                                <td
                                    class="px-4 py-3 text-sm text-gray-500 text-right"
                                    >{formatNumber(
                                        model.duration_hours,
                                        1,
                                    )}h</td
                                >
                            </tr>
                        {/each}
                    </tbody>
                </table>
            </div>
        </div>
    </Card>
{/if}
