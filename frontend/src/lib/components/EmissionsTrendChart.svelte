<script lang="ts">
    import { Chart, registerables } from "chart.js";
    import { onDestroy, onMount } from "svelte";

    Chart.register(...registerables);

    interface DataPoint {
        date: string;
        emissions_kg: number;
        energy_kwh: number;
        job_count: number;
        training_jobs: number;
        inference_jobs: number;
    }

    interface Props {
        dataPoints?: DataPoint[];
        title?: string;
        height?: number;
    }

    let {
        dataPoints = [],
        title = "Emissions Over Time",
        height = 300,
    }: Props = $props();

    let canvas: HTMLCanvasElement | null = $state(null);
    let chart: Chart | null = null;

    function createChart() {
        if (!canvas || dataPoints.length === 0) return;

        // Destroy existing chart
        if (chart) {
            chart.destroy();
        }

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        chart = new Chart(ctx, {
            type: "line",
            data: {
                labels: dataPoints.map((d) => d.date),
                datasets: [
                    {
                        label: "CO₂ Emissions (kg)",
                        data: dataPoints.map((d) => d.emissions_kg),
                        borderColor: "rgb(34, 197, 94)",
                        backgroundColor: "rgba(34, 197, 94, 0.1)",
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        yAxisID: "y",
                    },
                    {
                        label: "Energy (kWh)",
                        data: dataPoints.map((d) => d.energy_kwh),
                        borderColor: "rgb(59, 130, 246)",
                        backgroundColor: "rgba(59, 130, 246, 0.1)",
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        yAxisID: "y1",
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: "index",
                    intersect: false,
                },
                plugins: {
                    legend: {
                        display: true,
                        position: "top",
                    },
                    title: {
                        display: false,
                    },
                    tooltip: {
                        callbacks: {
                            afterBody: function (context) {
                                const idx = context[0].dataIndex;
                                const point = dataPoints[idx];
                                return [
                                    "",
                                    `Jobs: ${point.job_count}`,
                                    `  Training: ${point.training_jobs}`,
                                    `  Inference: ${point.inference_jobs}`,
                                ];
                            },
                        },
                    },
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: "Date",
                        },
                        grid: {
                            color: "rgba(0, 0, 0, 0.05)",
                        },
                    },
                    y: {
                        type: "linear",
                        display: true,
                        position: "left",
                        title: {
                            display: true,
                            text: "CO₂ (kg)",
                        },
                        grid: {
                            color: "rgba(0, 0, 0, 0.05)",
                        },
                    },
                    y1: {
                        type: "linear",
                        display: true,
                        position: "right",
                        title: {
                            display: true,
                            text: "Energy (kWh)",
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    },
                },
            },
        });
    }

    // Update chart when data changes
    $effect(() => {
        if (canvas && dataPoints.length > 0) {
            createChart();
        }
    });

    onMount(() => {
        if (canvas && dataPoints.length > 0) {
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

    {#if dataPoints.length === 0}
        <div
            class="flex items-center justify-center h-64 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300"
        >
            <p class="text-gray-500">No trend data available yet</p>
        </div>
    {:else}
        <div class="relative" style="height: {height}px;">
            <canvas bind:this={canvas}></canvas>
        </div>
    {/if}
</div>
