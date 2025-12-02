<script lang="ts">
    import { Chart, registerables } from "chart.js";
    import { onDestroy, onMount } from "svelte";

    Chart.register(...registerables);

    interface ModelComparison {
        model: string;
        emissions_kg: number;
        energy_kwh: number;
        duration_hours: number;
        job_count: number;
        avg_emissions_per_job: number;
    }

    interface Props {
        data?: ModelComparison[];
        title?: string;
        height?: number;
    }

    let {
        data = [],
        title = "Emissions by Model",
        height = 300,
    }: Props = $props();

    let canvas: HTMLCanvasElement | null = $state(null);
    let chart: Chart | null = null;

    // Generate colors for models
    function getColors(count: number): { bg: string[]; border: string[] } {
        const hues = Array.from({ length: count }, (_, i) => (i * 137.5) % 360);
        return {
            bg: hues.map((h) => `hsla(${h}, 70%, 60%, 0.7)`),
            border: hues.map((h) => `hsl(${h}, 70%, 50%)`),
        };
    }

    function createChart() {
        if (!canvas || data.length === 0) return;

        if (chart) {
            chart.destroy();
        }

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const colors = getColors(data.length);

        chart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: data.map((d) =>
                    d.model.length > 20
                        ? d.model.substring(0, 20) + "..."
                        : d.model,
                ),
                datasets: [
                    {
                        label: "Total CO₂ (kg)",
                        data: data.map((d) => d.emissions_kg),
                        backgroundColor: colors.bg,
                        borderColor: colors.border,
                        borderWidth: 1,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: "y",
                plugins: {
                    legend: {
                        display: false,
                    },
                    tooltip: {
                        callbacks: {
                            afterLabel: function (context) {
                                const idx = context.dataIndex;
                                const model = data[idx];
                                return [
                                    `Jobs: ${model.job_count}`,
                                    `Avg per job: ${model.avg_emissions_per_job.toFixed(4)} kg`,
                                    `Total energy: ${model.energy_kwh.toFixed(2)} kWh`,
                                    `Duration: ${model.duration_hours.toFixed(1)}h`,
                                ];
                            },
                        },
                    },
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "CO₂ Emissions (kg)",
                        },
                        grid: {
                            color: "rgba(0, 0, 0, 0.05)",
                        },
                    },
                    y: {
                        grid: {
                            display: false,
                        },
                    },
                },
            },
        });
    }

    $effect(() => {
        if (canvas && data.length > 0) {
            createChart();
        }
    });

    onMount(() => {
        if (canvas && data.length > 0) {
            createChart();
        }
    });

    onDestroy(() => {
        if (chart) {
            chart.destroy();
        }
    });
</script>

<div class="w-full">
    <h3 class="text-lg font-semibold text-gray-900 mb-4">{title}</h3>

    {#if data.length === 0}
        <div
            class="flex items-center justify-center h-64 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300"
        >
            <p class="text-gray-500">No model comparison data available</p>
        </div>
    {:else}
        <div
            class="relative"
            style="height: {Math.max(height, data.length * 40)}px;"
        >
            <canvas bind:this={canvas}></canvas>
        </div>
    {/if}
</div>
