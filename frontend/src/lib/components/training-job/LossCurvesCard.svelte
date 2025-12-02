<script lang="ts">
    import Card from "$lib/components/Card.svelte";
    import LossChart from "$lib/components/LossChart.svelte";

    interface Props {
        trainingMetrics: any[];
        validationMetrics: any[];
    }

    let { trainingMetrics, validationMetrics }: Props = $props();
</script>

{#if trainingMetrics.length > 0 || validationMetrics.length > 0}
    <Card>
        <div class="p-6">
            <LossChart
                {trainingMetrics}
                {validationMetrics}
                title="Training & Validation Loss"
                height={350}
            />

            <!-- Metrics Table -->
            <div class="mt-6">
                <h4 class="text-sm font-semibold text-gray-900 mb-3">
                    Recent Metrics
                </h4>
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50">
                            <tr>
                                <th
                                    scope="col"
                                    class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                                >
                                    Step
                                </th>
                                <th
                                    scope="col"
                                    class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                                >
                                    Training Loss
                                </th>
                                {#if validationMetrics.length > 0}
                                    <th
                                        scope="col"
                                        class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                                    >
                                        Validation Loss
                                    </th>
                                {/if}
                                <th
                                    scope="col"
                                    class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                                >
                                    Learning Rate
                                </th>
                            </tr>
                        </thead>
                        <tbody class="bg-white divide-y divide-gray-200">
                            {#each trainingMetrics
                                .slice(-10)
                                .reverse() as metric}
                                <tr>
                                    <td
                                        class="px-3 py-2 whitespace-nowrap text-sm text-gray-900"
                                    >
                                        {metric.step}
                                    </td>
                                    <td
                                        class="px-3 py-2 whitespace-nowrap text-sm text-gray-900"
                                    >
                                        {metric.loss.toFixed(4)}
                                    </td>
                                    {#if validationMetrics.length > 0}
                                        <td
                                            class="px-3 py-2 whitespace-nowrap text-sm text-gray-900"
                                        >
                                            {#if validationMetrics.find((v) => v.step === metric.step)}
                                                {validationMetrics
                                                    .find(
                                                        (v) =>
                                                            v.step ===
                                                            metric.step,
                                                    )
                                                    ?.loss.toFixed(4)}
                                            {:else}
                                                -
                                            {/if}
                                        </td>
                                    {/if}
                                    <td
                                        class="px-3 py-2 whitespace-nowrap text-sm text-gray-900"
                                    >
                                        {metric.learning_rate
                                            ? metric.learning_rate.toExponential(
                                                  2,
                                              )
                                            : "-"}
                                    </td>
                                </tr>
                            {/each}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </Card>
{/if}
