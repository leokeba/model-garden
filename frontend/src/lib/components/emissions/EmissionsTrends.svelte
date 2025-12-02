<script lang="ts">
    import Card from "$lib/components/Card.svelte";
    import EmissionsTrendChart from "$lib/components/EmissionsTrendChart.svelte";

    interface TrendDataPoint {
        date: string;
        emissions_kg: number;
        energy_kwh: number;
        job_count: number;
        training_jobs: number;
        inference_jobs: number;
    }

    interface Props {
        trendData: TrendDataPoint[];
        trendPeriod: "7d" | "30d" | "90d" | "all";
        loading: boolean;
        onPeriodChange: (period: "7d" | "30d" | "90d" | "all") => void;
    }

    let { trendData, trendPeriod, loading, onPeriodChange }: Props = $props();

    function formatNumber(num: number, decimals = 2): string {
        return num.toFixed(decimals);
    }

    const periodOptions = [
        { value: "7d", label: "7 Days" },
        { value: "30d", label: "30 Days" },
        { value: "90d", label: "90 Days" },
        { value: "all", label: "All Time" },
    ] as const;
</script>

<Card class="mb-6">
    <div class="p-4 border-b border-gray-200">
        <div class="flex items-center gap-4">
            <span class="text-sm font-medium text-gray-700">Time Period:</span>
            <div class="flex gap-2">
                {#each periodOptions as option}
                    <button
                        onclick={() => onPeriodChange(option.value)}
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
        {#if loading}
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
